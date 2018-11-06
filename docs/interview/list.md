链表
====
链表是一种线性表，但是在物理存储上并不是线性存储的（即不是连续的内存空间，这样可以更充分利用内存碎片）；而是在每一个元素中，存储了下一个元素的地址。这样相比数组，会多占用一些空间<br>
插入和删除操作效率非常高，查询比较慢。

## 单向链表
每个结点包含下一个结点的指针；<br>
单链表基本结构，包含插入、删除、翻转基本操作：
```Java
    static class SinglyList {
        SinglyListNode head;
        int size;

        /**
         * 为一个链表增加结点，增加的结点的位置为k，k从1开始
         * 即当k=1时，新增结点为头结点
         *
         * @param data
         * @param k
         */
        public void addAsK(int data, int k) {
            if (k > this.size + 1 || k < 1) {
                throw new RuntimeException("k value error! k=" + k);
            }
            SinglyListNode pre = null;
            while (k-- > 1) {
                if (pre == null) {
                    pre = this.head;
                } else {
                    pre = pre.next;
                }
            }
            SinglyListNode newNode = new SinglyListNode(data);
            if (pre == null) {
                newNode.next = this.head;
                this.head = newNode;
            } else {
                newNode.next = pre.next;
                pre.next = newNode;
            }
            size++;
        }

        /**
         * 删除第K个结点，K从1开始
         *
         * @param k
         * @return
         */
        public int deleteHead(int k) {
            if (k < 1 || k > this.size) {
                throw new RuntimeException("k value error! k=" + k);
            }
            SinglyListNode preNode = null;
            while (k-- > 1) {
                if (preNode == null) {
                    preNode = this.head;
                } else {
                    preNode = preNode.next;
                }
            }
            SinglyListNode delNode;
            if (preNode == null) {
                delNode = this.head;
                this.head = this.head.next;
            } else {
                delNode = preNode.next;
                preNode.next = delNode.next;
            }
            size--;
            return delNode.data;
        }

        public void reverse() {
            SinglyListNode currentNode = head;
            SinglyListNode nextNode = head.next;
            head.next = null;
            SinglyListNode tmpNode;
            while (nextNode != null) {
                tmpNode = nextNode.next;
                nextNode.next = currentNode;
                currentNode = nextNode;
                nextNode = tmpNode;
            }
            this.head = currentNode;
        }

        @Override
        public String toString() {
            StringBuilder stringBuilder = new StringBuilder("size=").append(size).append("\tvalues:");
            SinglyListNode node = head;
            while (node != null) {
                stringBuilder.append(node.data).append(",");
                node = node.next;
            }
            return stringBuilder.toString();
        }

        private static class SinglyListNode {
            int data;
            SinglyListNode next;

            public SinglyListNode(int data) {
                this.data = data;
            }
        }
    }
```

## 双向链表
每个结点有两个指针，分别指向其前驱结点和后继结点。<br>
插入结点：
```
```
删除结点：
```
```
链表翻转：
```
```
