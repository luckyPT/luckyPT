MySql
====
### 乐观锁与悲观锁
乐观锁是指在执行事务期间不加锁，提交的时候先进行版本校验，如果数据发生变更则回滚本次事务，如果没有变更则提交事务。在关系型数据库中，通常通过增加version字段记录数据版本（可以是整形或者时间戳），在提交事务的时候限定只有当数据版本与读取数据时一致才更新;
>select cols_a,cols_b,version from table_name where...;一系列处理; update table_name set ... ... where version = old_version;

悲观锁在执行事务期间始终加锁，防止其他事务修改数据。

