package com.ysz.biz.hbase;

import static org.apache.hadoop.hbase.util.Bytes.toBytes;
import static org.apache.hadoop.hbase.util.Bytes.toInt;

import org.apache.hadoop.hbase.TableName;
import org.apache.hadoop.hbase.client.Connection;
import org.apache.hadoop.hbase.client.Get;
import org.apache.hadoop.hbase.client.Result;
import org.apache.hadoop.hbase.client.Table;

public class HBaseDm_001 {

  public static void main(String[] args) throws Exception {
    HBaseConnectionBean hBaseConnectionBean = new HBaseConnectionBean();
    hBaseConnectionBean.setHbaseZk("10.1.3.52,10.1.3.32,10.1.3.51");
    hBaseConnectionBean.afterPropertiesSet();

    final String normalizedBlogId = HBaseConnectionBean.normalize("1006890568");
    System.err.println(normalizedBlogId);
    try (Connection connection = hBaseConnectionBean.getConnection()) {

      try (Table table = connection.getTable(TableName.valueOf("mblog_like_counter"))) {
        final byte[] rowkey = toBytes(normalizedBlogId);
        final byte[] columnFam = toBytes("c");
        Get get = new Get(rowkey);
        Result result = table.get(get);
        System.err.println(toInt(result.getValue(columnFam, toBytes("likeCnt"))));
//        System.err.println(FastDateFormat.getInstance() .format(toLong(result.getValue(columnFam, toBytes("favAt")))));
      }
    }

  }
}
