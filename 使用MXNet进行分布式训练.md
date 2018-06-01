# 使用MXNet进行分布式训练

MXNet支持分布式训练，使我们能够利用多台机器进行更快速的训练。 在本文中，我们描述它是如何工作的，如何启动分布式训练工作和一些提供更多控制的环境变量。

##  并行类型

我们可以通过两种方式来将训练神经网络的工作量分配在多个设备（可以是GPU或CPU）。 第一种方式是数据并行，指的是每个设备存储完整模型副本的情况。 每个设备使用数据集的不同部分进行工作，设备共同更新共享模型。 这些设备可以位于一台机器上，也可以位于多台机器上。 在本文中，我们将描述如何以数据并行方式训练一个模型，其中设备分布在多台机器上。
当模型太大以至于不能适配设备内存时，第二种称为模型并行的方法就很有用。 在这里，不同的设备被分配了学习模型不同部分的任务。 目前，MXNet仅支持单机中的模型并行。 有关这方面的更多信息，请参考[使用模型并行的多GPU的培训](https://mxnet.incubator.apache.org/versions/master/faq/model_parallel_lstm.html)。

## 如何进行分布式训练

接下来的几个概念是理解使用MXNet进行分布式训练的关键：

### 进程的类型

MXNet中有三种进程类型，这些进程之间相互通信，完成模型的训练。

* Worker：Worker节点实际上在一批训练样本上进行训练。 在处理每个批次之前，Workers从服务器上拉出权重。 Worker还会在每批次处理后向服务器发送梯度(gradient)。 根据训练模型的工作量，在同一台机器上运行多个工作进程可能不是一个好主意。

* Server：可以有多个Servers存储模型的参数，并与Workers进行交流。 Serverd可能与工作进程同处一处,也可能不。

* Scheduler(调度器)：只有一个Scheduler。Scheduler的作用是配置集群。这包括等待每个节点启动以及节点正在监听哪个端口之类的消息。 然后Scheduler让所有进程知道集群中的其他节点的信息，以便它们可以相互通信。

### kvstore

MXNet提供了key-value存储机制，这个机制是多设备训练中关键的一部分。一个或多个Server通过将参数存储为K-V的形式在单台机器或者多台机器上进行跨节点的参数交互。这种存储机制中的每个值都由key-value表示，其中网络中的每个参数数组被分配了一个key,并且value是这个参数数组的权重。Workes在一批计算处理之后进行梯度的推送，并且在新的计算批次开始之前拉取更新后的权重。我们也可以在更新每个权重时传入K-V存储的优化器。这个优化器像随机梯度下降一样定义了一个更新规则——本质上是旧的权重、梯度和一些参数来计算新的权重。

如果你使用一个Gluon Trainer对象或者是模型的API,它将在内部使用kvstore对象来聚合梯度，这些梯度来自同一台机器上或者不同机器上多个设备上。

尽管无论是否使用多台机器进行训练，API都保持不变，但kvstore服务器的概念仅存在于分布式训练期间。在分布式情况下，每次推送和拉取都涉及与kvstore服务器的通信。当一台机器上有多个设备时，这些设备训练的梯度首先会聚合在机器上，然后发送再到服务器。
请注意，我们需要构建标志`USE_DIST_KVSTORE = 1`之后再编译MXNet才能使用分布式训练机制。

通过调用`mxnet.kvstore.create`函数使用包含dist字符串的字符串参数来启用KVStore的分布式模式，如下所示：

> kv = mxnet.kvstore.create('dist_sync')

有关KVStore的更多信息，请参阅[KVStore API](https://mxnet.incubator.apache.org/versions/master/api/python/kvstore/kvstore.html)。

### Keys的分配

每个server不一定存储所有的key或全部的参数数组。 参数分布在不同的server上。 哪个server存储特定的keys是随机决定的。 KVStore透明地处理不同服务器上的keys分配。 它确保当一个keys被拉取时，该请求被发送到的服务器具有对应value。 如果某个keys的值非常大，则可能会在不同的服务器上分片。 这意味着不同的服务器拥有不同部分的value。 并且，这个处理是透明的，所以workers不必做任何不同的事情。 这个分片的阈值可以用环境变量`MXNET_KVSTORE_BIGARRAY_BOUND`来控制。 有关更多详情，请参阅[环境变量](https://github.com/apache/incubator-mxnet/blob/24362d0ce5d2d099dd65abc0ccd666f7a131d8e0/docs/faq/distributed_training.md#environment-variables)。

### 切分训练数据

在数据并行模式下进行分布式训练时，我们希望每台机器都在不同部分的数据集上工作。

对于单个worker的数据并行训练，我们可以使用`mxnet.gluon.utils.split_and_load`来切分数据迭代器(data iterator)提供的一批样本，然后将该批处理的每个部分加载到将处理它的设备上。

在分布式训练的情况下，我们需要在开始时将数据集分成`n`个部分，以便每个worker获得不同的部分。然后，每个worker可以使用`split_and_load`再次将数据集的这部分划分到单个机器上的不同设备上。

通常情况下，每个worker都是通过数据迭代器进行的数据拆分，通过传递切分的数量和切分部分的索引来迭代。 MXNet中支持此功能的一些迭代器是`mxnet.io.MNISTIterator`和`mxnet.io.ImageRecordIter`。如果你使用的是不同的迭代器，你可以看看上面的迭代器是如何实现此功能的。我们可以使用kvstore对象来获取当前worker的数量（kv.num_workers）和等级（kv.rank）。这些可以作为参数传递给迭代器。你可以看[example / gluon / image_classification.py](https://github.com/apache/incubator-mxnet/blob/master/example/gluon/image_classification.py)来查看一个示例用法。

### 分布式训练的不同模式

在kvstore创建包含dist字段的字符串时才启用分布式。

通过使用不同类型的kvstore可以启用不同的分布式训练模式。

* `dist_sync` : 在同步分布式训练中，所有worker在每批计算开始时都使用同一组同步模型参数。这意味着每次批处理后，服务器在更新模型参数之前都会等待从每个worker上接收gradients。这种同步需要付出代价，因为worker必须等到服务器完成接收过程再开始拉取参数。在这种模式下，如果有worker崩溃，那么它会使所有工人的进度停止。

* `dist_async` : 在异步分布式训练中，server从一名worker处接收梯度之后，立即更新其存储，以用于响应任何未来的拉取。这意味着完成一批计算的工作人员可以从server中提取当前参数并开始下一批计算，即使其他工作人员尚未完成先前批的计算。这比`dist_sync`快，但可能需要更多的训练次数才能收敛。在异步模式下，需要传递优化器，因为在没有优化器的情况下，kvstore会用接收的权重替换存储的权重，这对于异步模式下的训练没有意义。权重的更新具有原子性，这意味着同一重量不会同时发生两次更新。但是，更新顺序无法保证。

* `dist_sync_device`: 与dist_sync相同，当每个节点上使用多个GPU时使用，此模式在GPU上聚合梯度并更新权重，而`dist_sync`则在CPU内存上执行此类操作。此模式比dist_sync快，因为它可以减少GPU和CPU之间昂贵的通信，但会增加GPU上的内存使用量。

* `dist_async_device` ：与`dist_sync_device`相似，但处于异步模式。

### 梯度压缩

当通信费用昂贵，并且计算时间与通信时间的比例较低时，通信可能成为瓶颈。 在这种情况下，可以使用梯度压缩来降低通信成本，从而加速训练。 有关更多详细信息，请参阅[梯度压缩](https://mxnet.incubator.apache.org/versions/master/faq/gradient_compression.html)。

注意：对于小型模型，当计算成本远低于通信成本时，由于通信和同步的开销，分布式培训实际上可能比单台机器上的培训慢。

## 如何开始分布式训练

MXNet提供了一个脚本工具/ launch.py，以便于开展分布式训练工作。这支持各种类型的集群资源管理器，如`ssh`，`mpirun`，`yarn`和`sge`。 如果您已配置了其中一个集群，则可以跳过下一节设置群集。 如果您想使用上述未提及的类型，请直接跳到手动启动作业部分。

### 配置集群

使用[AWS CloudFormation template](https://github.com/awslabs/deeplearning-cfn)配置用于分布式深度学习的EC2实例集群的是一个简单的方法。 如果您不能使用上述内容，本节将帮助您手动设置一组实例，以使您可以使用`ssh`启动分布式训练作业。 让我们用一台机器作为集群的`master`，我们将通过它启动并监视所有机器上的分布式培训。

如果集群中的计算机是AWS EC2等云计算平台的一部分，那么您的实例应该已经使用基于密钥的身份验证。 确保使用相同的密钥创建所有实例，例如使用`mxnet-key`并且所有实例位于同一个安全组中。 接下来，我们需要确保master能够通过`ssh`访问集群中其他所有机器。方法是将此密钥添加到[ssh-agent](https://en.wikipedia.org/wiki/Ssh-agent)并在登录时将其转发给master。这将使mxnet-key成为master上的默认密钥:

```
ssh-add .ssh/mxnet-key
ssh -A user@MASTER_IP_ADDRESS
```
如果您的机器使用密码进行身份验证，请参阅[此处](https://help.ubuntu.com/community/SSH/OpenSSH/Keys)获取有关在机器之间设置无密码身份验证的说明。

如果所有这些机器都具有共享的文件系统，以便他们可以访问培训脚本，则会更简便。 一种方法是使用Amazon Elastic File System来创建您的网络文件系统。 安装AWS Elastic File System时，以下命令中的选项是推荐的选项。

```
sudo mkdir efs && sudo mount -t nfs4 -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2 NETWORK_FILE_SYSTEM_IP:/ efs
```
提示：您可能会发现将大型数据集存储在S3上有助于集群中所有机器的轻松访问。 请参阅[使用S3的数据进行训练](https://mxnet.incubator.apache.org/versions/master/faq/s3_integration.html)以获取更多信息。

### 使用 Launch.py

MXNet提供了一个脚本[tools/launch.py](https://github.com/apache/incubator-mxnet/blob/master/tools/launch.py)，以便使用`ssh`，`mpi`，`sge`或`yarn`在集群上启动分布式培训。 您可以通过克隆mxnet仓库来获取此脚本。
```
git clone --recursive https://github.com/apache/incubator-mxnet
```

#### Example

让我们考虑使用[example/gluon/image_classification.py](https://github.com/apache/incubator-mxnet/blob/master/example/gluon/image_classification.py)在CIFAR10数据集上训练VGG11模型。
```
cd example/gluon/
```

在单台机器上，我们可以运行这个脚本，如下所示：

```
python image_classification.py --dataset cifar10 --model vgg11 --num-epochs 1
```

对于此示例的分布式培训，我们将执行以下操作：
如果包含脚本image_classification.py的mxnet目录可供集群中的所有计算机访问（例如，如果它们位于网络文件系统上），则可以运行：
```
../../tools/launch.py -n 3 -H hosts --launcher ssh python image_classification.py --dataset cifar10 --model vgg11 --num-epochs 1 --kvstore dist_sync
```

如果包含脚本的目录不能从集群中的其他机器访问，那么我们可以将当前目录同步到所有机器。

```
../../tools/launch.py -n 3 -H hosts --launcher ssh --sync-dst-dir /tmp/mxnet_job/ python image_classification.py --dataset cifar10 --model vgg11 --num-epochs 1 --kvstore dist_sync
```
> 如果您没有准备好集群但是仍想尝试此操作，请传递选项`--launcher local`而不是`ssh`

#### 选项
在这里，launch.py用于提交分布式训练作业。它有以下选择：

* `-n` 表示要启动的worker节点的数量。

* `-s` 表示要启动的server节点的数量。 如果没有指定，则认为它等于worker节点的数量。该脚本尝试循环访问hosts文件以启动server和worker。 例如，如果主机文件中有5个主机，并且您将n设置为3（并且不设置s）。 该脚本将启动总共3个server进程，前三台主机分别启动一个worker进程，总共3个worker进程.启动server进程的分别为第四台，第五台和第一台主机。如果主机文件中恰好有n个工作节点，它将在每台主机上启动一个服务器进程和一个工作进程。

* `--launcher` 表示通信模式。选项有：
  * `ssh` 如果机器可以通过ssh进行通信而无需密码。 这是默认启动模式。
  * `mpi` 使用Open MPI时开启
  * `sge` 适用于Sun Grid引擎
  * `yarn` 适用于Apache yarn
  * `local` 用于在同一本地计算机上启动所有进程。 这可以用于调试。


* `-H` 需要主机文件的路径,该文件包含集群中机器的IP。这些机器应能够在不使用密码的情况下相互通信。 此文件仅适用于启动程序模式为ssh或mpi时。 hosts文件内容的例子如下所示：
```
172.30.0.172
172.31.0.173
172.30.1.174
```

* `--sync-dst-dir` 将所有主机上的一个目录的路径指向当前将被同步的工作目录。此选项仅支持`ssh`启动模式。 当工作目录不能被群集中的所有机器访问时，这是必需的。设置此选项可在作业启动之前使用rsync同步当前目录。如果您尚未在系统范围内安装MXNet，则必须在运行launch.py之前将文件夹`python/mxnet`和文件`lib/libmxnet.so`复制到当前目录中。 例如，如果你在`example/gluon`中，你可以用`cp -r ../../python/mxnet../../lib/libmxnet.so`来做到这一点。如果你的`lib`文件夹中包含`libmxnet.so`，这将有效。 所以，就像你使用make的情况一样。 如果你使用CMake，这个文件将在你的`build`目录中。

* `python image_classification.py --dataset cifar10 --model vgg11 --num-epochs 1 --kvstore dist_sync`是每台机器上的训练工作的命令。请注意使用脚本中的`dist_sync`设置kvstore。

#### 终止工作

如果训练作业因错误而崩溃，或者如果我们试图在训练运行时终止启动脚本，则所有机器上的作业可能没有终止。 在这种情况下，我们需要手动终止它们。 如果我们使用的是ssh启动器，可以通过运行以下命令来完成，其中hosts是hostfile的路径。
```
while read -u 10 host; do ssh -o "StrictHostKeyChecking no" $host "pkill -f python" ; done 10<hosts
```

### 手动启动工作

如果由于某种原因，您不想使用上面的脚本启动分布式培训，那么本节将有所帮助。 MXNet使用环境变量将不同的角色分配给不同的进程，并让不同的进程查找调度程序。 需要按照以下步骤正确设置环境变量才能开始培训：

* `DMLC_ROLE` ：指定进程的角色。 这可以是server、worker或scheduler。 请注意，应该只有一个scheduler。 当`DMLC_ROLE`设置为server或scheduler后，这些进程在导入mxnet时启动。

* `DMLC_PS_ROOT_URI` ：指定scheduler的IP

* `DMLC_PS_ROOT_PORT` ：指定scheduler侦听的端口

* `DMLC_NUM_SERVER` ：指定群集中有多少个server节点

* `DMLC_NUM_WORKER` ：指定群集中有多少个worker节点

以下是在Linux或Mac上在单机启动所有作业的示例。 请注意，在同一台机器上启动所有作业不是一个好主意。 这只是为了使用法清楚展示。

```
export COMMAND=python example/gluon/mnist.py --dataset cifar10 --model vgg11 --num-epochs 1 --kv-store dist_async
DMLC_ROLE=server DMLC_PS_ROOT_URI=127.0.0.1 DMLC_PS_ROOT_PORT=9092 DMLC_NUM_SERVER=2 DMLC_NUM_WORKER=2 COMMAND &
DMLC_ROLE=server DMLC_PS_ROOT_URI=127.0.0.1 DMLC_PS_ROOT_PORT=9092 DMLC_NUM_SERVER=2 DMLC_NUM_WORKER=2 COMMAND &
DMLC_ROLE=scheduler DMLC_PS_ROOT_URI=127.0.0.1 DMLC_PS_ROOT_PORT=9092 DMLC_NUM_SERVER=2 DMLC_NUM_WORKER=2 COMMAND &
DMLC_ROLE=worker DMLC_PS_ROOT_URI=127.0.0.1 DMLC_PS_ROOT_PORT=9092 DMLC_NUM_SERVER=2 DMLC_NUM_WORKER=2 COMMAND &
DMLC_ROLE=worker DMLC_PS_ROOT_URI=127.0.0.1 DMLC_PS_ROOT_PORT=9092 DMLC_NUM_SERVER=2 DMLC_NUM_WORKER=2 COMMAND
```
有关scheduler如何设置群集的深入讨论，请参阅[此处](https://blog.kovalevskyi.com/mxnet-distributed-training-explained-in-depth-part-1-b90c84bda725)。

## 环境变量

### 关于调整性能

* `MXNET_KVSTORE_REDUCTION_NTHREADS` 数值类型：Integer； 默认值：4 (用于在单台计算机上汇总大型数组的CPU线程数);此函数也将用于`dist_sync` kvstore，以在单台计算机上汇总来自不同环境下的数组。使用`dist_sync_device`kvstore汇总数组也不会受到GPU上的影响。

* `MXNET_KVSTORE_BIGARRAY_BOUND` 数值类型：Integer；默认值：1000000 (大数组的最小规模)；当数组大小大于此阈值时，`MXNET_KVSTORE_REDUCTION_NTHREADS`线程用于减少数组规模。该参数也用作kvstore中的负载均衡器。它控制何时将单个权重拆分给所有server。如果单个权重矩阵的规模小于这个界限，那么它将被发送到一个随机挑选的server;否则，它被拆分到所有的服务器。

* `MXNET_ENABLE_GPU_P2P` GPU对等(P2P)通信 数值类型：boolean (0-false或1-true）；默认值：1（true）；如果为真，MXNet会尝试使用GPU对等通信（如果设备上可用）。 这仅在kvstore中包含类型设备时使用。

### 通信

* `DMLC_INTERFACE` 使用特定网络接口 数值类型：端口的名称 例如：`eth0` MXNet通常选择第一个可用网络接口。 但对于具有多个接口的机器，我们可以使用此环境变量指定要使用哪个网络接口进行数据通信。

* `PS_VERBOSE`  记录通信 数值类型：1或2；默认值：（空）
  * `PS_VERBOSE=1` 记录连接信息，如所有节点的IP和端口
  * `PS_VERBOSE=2` 记录所有数据通信信息

当网络不可靠时，从一个节点发送到另一个节点的消息可能会丢失。当关键的消息没有成功传递时，训练过程可能会挂起。 在这种情况下，可以为每个消息发送额外的ACK以跟踪其传送。这可以通过设置`PS_RESEND`和`PS_RESEND_TIMEOUT`来完成

* `PS_RESEND` 重传不可靠的网络 数值类型：boolean (0-false或1-true）；默认值：0（false）;是否启用重传消息

* `PS_RESEND_TIMEOUT` 收到ACK的超时 数值类型：Integer (in milliseconds)；默认值：1000；如果在`PS_RESEND_TIMEOUT`毫秒内未收到ACK，则该消息将被重发。


* `PS_RESEND`

* `PS_RESEND_TIMEOUT`
