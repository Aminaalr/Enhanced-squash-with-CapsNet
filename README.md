# Enhanced-squash-with-CapsNet
Introduction:
Capsule networks (CapsNet) have shown promise in various machine learning tasks due to their ability to capture hierarchical relationships and spatial hierarchies within data. The squash function, a crucial component of CapsNet, transforms capsule activations. Despite the success of standard squash functions, they face limitations such as vanishing gradients and difficulty in learning complex patterns with small vectors, especially in large-scale datasets.

Objective:
Building upon previous research recommending enhancements to squash functions, this project aims to improve CapsNet performance, particularly in high-dimensional and intricate data scenarios, such as bone marrow cell (BM) classification.

Methodology:
We extend our methodology to enhance the squash function to address the challenges faced by standard squash functions. The enhanced squash function aims to mitigate information loss during the routing process and improve feature representation, preserving spatial relationships.

Results:
Implementing the enhanced squash function yielded promising results, particularly in BM cell classification. The classification accuracy on BM data increased from 96.99% to 98.52%, demonstrating the efficacy of our methodology in enhancing CapsNet performance for complex tasks.

Performance Comparison:
Comparing the performance of the improved CapsNet model with the standard CapsNet across different datasets reveals consistent superiority of the enhanced squash CapsNet. It achieved accuracy percentages of 99.93% on MNIST, 73% on CIFAR-10, and 94.66% on Fashion MNIST, outperforming the standard model.

Conclusion:
The enhanced squash function significantly improves CapsNet performance across diverse datasets, confirming its potential for real-world applications in machine learning tasks. These findings underscore the importance of further research in enhancing CapsNet methodologies, particularly in handling complex and large-scale data.
