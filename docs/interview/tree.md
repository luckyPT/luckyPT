树相关的算法题
====
1、翻转二叉树
```Code
//需要深入理解入参和返回的是一个结点，就能理解递归的翻转过程
mirrorTree(Node node){
  if(node = null) return null;
  Node left = mirrorTree(node.left);
  Node right = mirrorTree(node.right);
  root.left = right;
  root.right = left;
  return root;
}
```
