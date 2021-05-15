package com.ysz.biz.mysql.shard;

public class ShardDm {

    public static void main(String[] args) throws Exception {
        int tableNum = 32;

        for (int i = 0; i < 123; i++) {
            System.err.println(i % tableNum);
        }
    }
}
