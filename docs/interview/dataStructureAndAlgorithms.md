数据结构与算法
====
数据结构就是用来存储与组织数据的方式。数据集合需要借助一定的数据结构来存储。
## 数组
数组是存储在一块连续的内存空间内的同类元素的集合。具有如下优缺点：<br>
优点：<br>
&ensp;&ensp;根据位置查询元素效率高。<br>
缺点：<br>
&ensp;&ensp;删除和插入元素效率比较慢。<br>
&ensp;&ensp;数组大小确定之后就不能改变。<br>

### 数组排序

#### 快速排序
思路：以第一个元素为“标兵”，通过end和start指针分别向前向后移动（end先移动），保证end走过的路都大于“标兵”，start走过的路都大于“小于”，不满足条件时则start和end的位置元素进行交换，另一个指针移动，直到二者相遇。这样数组就会被分为两部分，前半部分的值都小于后半部分。针对前半部分和后半部分再分别重复上述过程，直到最终只剩下一个元素终止。

#### 插入排序
思路：假如有一个队伍需要按照大小个排序，现在除了最后一个人之外，其他的都已经按照大小个排好序了。如何为最后一个人找到合适的位置？让最后一个人一步一步向前走，如果发现前面的人比自己高，则让前面的人后退，直到前面的人比自己小或者相等时，就停止。<br>
对于一个无序数组，就是先从最前面的两个人的数组子序列开始排，排好之后排三个人的子序列，依次递增，直到增至全部元素。

#### 归并排序
**二路归并**<br>
思路：假如现在有两个已经排序好的队伍，需要将这两个队伍合并成一个队伍。首先找一个可以容纳所有人的场地（对应新建一个数组），然后以两个队伍的第一个元素为各自的指针，比较两个指针元素的值，值小的按照次序进入新队伍，并将相应指针后移。直到两个队伍的所有元素都进入新场地。<br>
**数组排序**<br>
借助二路归并排序的思路，对数组进行排序的过程如下：<br>
function1 的函数：接收一个数组，开始元素位置、结束元素位置以及中间元素，并保证被中间元素分割的前后两部分都是有序的。最终完成对包含开始、结束元素以及中间元素的排序。<br>
function2 函数：知道一个数组是无序的，那么先将前半部分排序好，再将后半部分排序好，最后再将整体排序。伪代码如下：<br>
```
function2(array,start,end){
  if(start>=end){
    return;
  }
  mid = (start+end)/2;
  function2(array,start,mid);
  function2(array,mid+1,end);
  function1(array,start,mid,end)
}
```

#### 冒泡排序
冒泡排序的思路是，从前向后依次遍历，遇见大的记录其位置和值，直到最后，将最后一个元素和遍历过程中最大元素互换位置。<br>
重复上述过程。

## 树
树是具有n个结点的有限集合，并且结点之间存在一定的层次关系与父子关系。
### 相关概念
结点：包含一个数据元素以及其子结点的引用的一个数据对象。<br>
结点的度：某个节点拥有子节点的数目<br>
树的度：各结点度的最大值<br>
树的深度：根节点为第一层，深度为1，其后递增。
叶结点：度为0的结点<br>
非叶子结点：度不为0的结点<br>
孩子&双亲：结点的子树称为该结点的孩子，该结点称为孩子的双亲。
兄弟：同一个双亲的孩子之间互称兄弟。
### 二叉树
二叉树：每个结点最多有两个子树，并且子树有左右之分，次序不能颠倒。
**满二叉树：**深度为k的树，有2^k - 1个结点的树<br>
**完全二叉树：**深度为k，有n个结点，当且仅当这n个结点的编号都与深度为k的满二叉树从1到n的结点一一对应。换句话说就是自上而下，自左而右，可以有的结点不能没有。<br>

#### 二叉树性质
① 第i层最多有2^(i-1)个结点。i从1开始计数<br>
② 深度为k的二叉树，最多有2^k - 1个结点。<br>
③ 自上而下，自左向右依次编号，从0开始；则第i个结点的两个子节点编号分别为2i+1 和 2i+2。其父节点为(i+1)/2 - 1

#### 二叉树遍历
三种遍历方式，先序、中序、后序指的是遍历根结点的次序。
先序遍历（深度优先）：先遍历根结点，然后是左结点、右结点；<br>
中序遍历：先遍历左节点，再遍历根结点，最后遍历右结点；<br>
后序遍历的顺序是：左节点、右结点、根结点。<br>
无论哪种遍历方式都是先左后右。

先序遍历思路：<br>
①使用递归实现，伪代码：
```
preOrderTraversal(Node root){
  if(root==null) return;
  visit(root.value);
  preOrderTraverse(root.left);
  preOrderTraverse(root.right);
}
```
②使用循环实现：<br>
```
preOrderTraversal(Node root){
  if(root==null) return;
  Stack<Node> stack=new Stack<>();
  stack.push(root);
  while(!stack.isEmpty()){
     Node node = stack.pop()
     visit(node)
     stack.push(node.right)
     stack.push(node.left)
  }
}
```

中序遍历思路：<br>
①使用递归实现，伪代码：
```
inOrderTraversal(Node root){
  if(root==null) return;
  preOrderTraverse(root.left);
  visit(root.value);
  preOrderTraverse(root.right);
}
```
②使用循环实现：<br>
```
inOrderTraversal(Node root){
  if(root==null) return;
  Stack<Node> stack=new Stack<>();
  Node tmpNode = root;
  while(tmpNode!=null || !stack.isEmpty()){//切记是“或”的关系
    if(tmpNode!=null){//左子树入栈
      stack.push(tmpNode)
      tmpNode = tmpNode.left;
    }else{//弹出一个元素，并将其右子树视为一棵树的根节点
      tmpNode = stack.pop()
      visit(tmpNode)
      tmpNode = tmpNode.right
    }
  }
}
```

后序遍历思路：<br>
①使用递归实现，伪代码：
```
postOrderTraversal(Node root){
  if(root==null) return;
  preOrderTraverse(root.left);
  preOrderTraverse(root.right);
  visit(root.value);
}
```
②使用循环实现：<br>
```
postOrderTraversal(Node root){
  if(root==null) return;
  
}
```

#### 堆 & 堆排序

#### 二叉查找树 & 树表查找算法

#### 平衡二叉树

#### 红黑树

#### 哈弗曼树

### 字典树
