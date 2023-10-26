# SNN基础知识-Wikipedia

## 一、定义

**Spiking neural networks** (**SNNs**) are [artificial neural networks](https://en.wikipedia.org/wiki/Artificial_neural_network) that more closely mimic natural neural networks.[[1\]](https://en.wikipedia.org/wiki/Spiking_neural_network#cite_note-Maas_1996-1) In addition to [neuronal](https://en.wikipedia.org/wiki/Artificial_neuron) and [synaptic](https://en.wikipedia.org/wiki/Electrical_synapse) state, SNNs incorporate the concept of time into their [operating model](https://en.wikipedia.org/wiki/Operating_Model). The idea is that [neurons](https://en.wikipedia.org/wiki/Artificial_neuron) in the SNN do not transmit information at each propagation cycle (as it happens with typical multi-layer [perceptron networks](https://en.wikipedia.org/wiki/Perceptron)), but rather transmit information only when a [membrane potential](https://en.wikipedia.org/wiki/Membrane_potential)—an intrinsic quality of the neuron related to its membrane electrical charge—reaches a specific value, called the threshold. When the membrane potential reaches the threshold, the neuron fires, and generates a signal that travels to other neurons which, in turn, increase or decrease their potentials in response to this signal. A neuron model that fires at the moment of threshold crossing is also called a [spiking neuron model](https://en.wikipedia.org/wiki/Spiking_neuron_model).[[2\]](https://en.wikipedia.org/wiki/Spiking_neural_network#cite_note-2)

**Spiking神经网络（SNN）**是一种更接近自然神经网络的人工神经网络。除了神经元和突触状态外，SNN还将**时间**的概念纳入其操作模型。其想法是，SNN中的神经元不会在每个传播周期传输信息（就像典型的多层感知器网络一样），而是只有当膜电位（*membrane potential*）——神经元的一种与膜电荷相关的内在质量——达到一个特定值时才传输信息，该值被称为阈值（*threshold*）。当膜电位达到阈值时，神经元就会启动，并产生一个信号，传递到其他神经元，这些神经元反过来会根据这个信号增加或减少电位。在跨过阈值时触发的神经元模型也称为尖峰神经元模型。

![](C:\Users\Or1ano\OneDrive\桌面\研究生\SNN\图片\Neuron_bio.png)

The most prominent spiking neuron model is the [leaky integrate-and-fire](https://en.wikipedia.org/wiki/Biological_neuron_model#Leaky_integrate-and-fire) model. In the integrate-and-fire model, the momentary activation level (modeled as a [differential equation](https://en.wikipedia.org/wiki/Differential_equation)) is normally considered to be the neuron's state, with incoming spikes pushing this value higher or lower, until the state eventually either decays or—if the firing threshold is reached—the neuron fires. After firing, the state variable is reset to a lower value.

最突出的尖峰神经元模型是***leaky integrate-and-fire***模型。在模型中，瞬时激活水平（建模为微分方程）通常被认为是神经元的状态，传入的尖峰将该值推高或推低，直到状态最终衰减，或者——如果达到激发阈值——神经元激发。点火后，状态变量会重置为较低的值。

Various decoding methods exist for interpreting the outgoing *[spike train](https://en.wikipedia.org/wiki/Spike_train)* as a real-value number, relying on either the frequency of spikes (rate-code), the time-to-first-spike after stimulation, or the interval between spikes.

存在各种解码方法，用于根据尖峰的频率（速率码）、刺激后第一次尖峰的时间或尖峰之间的间隔将输出尖峰序列解释为实数。



## 二、基础材料

大脑中的信息被表示为动作电位（神经元尖峰），它可以被分为尖峰序列，甚至是大脑活动的协调波。神经科学的一个基本问题是确定神经元是通过速率还是时间编码进行通信。时间编码表明，单个尖峰神经元可以取代*sigmoidal*神经网络上数百个隐藏的单元。

SNN在**连续域**而不是离散域中进行计算。其想法是，神经元可能**不会**在每次传播迭代中测试激活（就像典型的多层感知器网络中的情况一样），而是只有当它们的膜电位（*membrane potential*）达到一定值时才测试激活。当神经元被激活时，它会产生一个信号，传递给连接的神经元，从而提高或降低它们的膜电位。

在尖峰神经网络中，神经元的当前状态被定义为其膜电位（可能建模为微分方程）。输入脉冲使膜电位上升一段时间，然后逐渐下降。已经有编码方案来将这些脉冲序列解释为一个数字，同时考虑脉冲频率和脉冲间隔。可以建立基于脉冲产生时间的神经网络模型。利用**脉冲发生的确切时间**，神经网络可以利用更多的信息并提供更好的计算性能。

SNN方法产生连续输出，而不是传统ANN的二进制输出。脉冲串不容易解释，因此需要如上所述的编码方案。然而，脉冲串表示可能更适合处理时空数据（或连续的真实世界感官数据分类）。SNN通过仅将神经元连接到附近的神经元来考虑空间，从而单独处理输入块（类似于使用滤波器的CNN）。他们通过将信息编码为脉冲串来考虑时间，以免在二进制编码中丢失信息。这避免了递归神经网络（RNN）的额外复杂性。事实证明，脉冲神经元是比传统人工神经元更强大的计算单元。

SNN 理论上比之前运用的神经网络更强大；然而，SNN 训练问题和硬件要求限制了它们的使用。尽管无监督的生物学启发学习方法可用，例如 Hebbian 学习和 STDP，但**没有**有效的监督训练方法适合 SNN，能够提供比之前运用的网络更好的性能。基于 Spike 的 SNN 激活是**不可微分**的，因此很难开发基于梯度下降的训练方法来执行误差反向传播，尽管 NormAD和多层 NormAD 等最近的一些算法已经通过基于尖峰激活的梯度的适当近似证明了良好的训练性能。

目前，研究人员正在积极研究使用 SNN 时遇到的一些挑战。**第一个挑战**涉及尖峰非线性的不可微性。前向和后向学习方法的表达式都包含神经激活函数的导数，该导数是不可微的，因为神经元的输出在尖峰时为 1，否则为 0。**二元尖峰非线性的这种全有或全无行为会阻止梯度“流动”**，并使 LIF 神经元不适合基于梯度的优化。**第二个挑战**涉及**优化算法**本身的实现。标准 BP 在计算、内存和通信方面可能非常昂贵，并且可能不太适合实现它的硬件（例如计算机、大脑或神经形态设备）所规定的约束。 关于第一个挑战，有几种方法可以克服它。

1. resorting to entirely biologically inspired local learning rules for the hidden units
2. translating conventionally trained “rate-based” NNs to SNNs
3. smoothing the network model to be continuously differentiable
4. defining an SG (Surogate Gradient) as a continuous relaxation of the real gradients



1. 针对隐藏单元采用完全受生物学启发的局部学习规则
2. 将传统训练的“基于速率”的神经网络转换为 SNN
3. 平滑网络模型以使其连续可微
4. 定义一个***SG***(*Surrogate Gradient*) 作为真实梯度的*continuous [relaxation](https://en.wikipedia.org/wiki/Relaxation_(approximation))*

