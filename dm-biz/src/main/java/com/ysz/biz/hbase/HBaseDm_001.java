package com.ysz.biz.hbase;

import static org.apache.hadoop.hbase.util.Bytes.toBytes;
import static org.apache.hadoop.hbase.util.Bytes.toInt;

import java.util.ArrayList;
import java.util.List;
import org.apache.hadoop.hbase.TableName;
import org.apache.hadoop.hbase.client.Connection;
import org.apache.hadoop.hbase.client.Get;
import org.apache.hadoop.hbase.client.Put;
import org.apache.hadoop.hbase.client.Result;
import org.apache.hadoop.hbase.client.Row;
import org.apache.hadoop.hbase.client.Table;

public class HBaseDm_001 {

  public static void main(String[] args) throws Exception {
    HBaseConnectionBean hBaseConnectionBean = new HBaseConnectionBean();
    hBaseConnectionBean.setHbaseZk("10.1.3.52,10.1.3.32,10.1.3.51");
    hBaseConnectionBean.afterPropertiesSet();

    try (Connection connection = hBaseConnectionBean.getConnection()) {

      try (Table table = connection.getTable(TableName.valueOf("message_message"))) {
        List<Row> batch = new ArrayList<>();
        final String normalizedBlogId = HBaseConnectionBean.normalize("1234");
        final byte[] rowkey = toBytes(normalizedBlogId);

        Put put = new Put(rowkey);
        final byte[] columnFam = toBytes("c");
        put.addColumn(columnFam, toBytes("status"), toBytes(0));
        batch.add(put);
        Object[] resArray = new Object[batch.size()];
        table.batch(batch, resArray);

        Get get = new Get(rowkey);
        Result result = table.get(get);
        byte[] rootId = result.getValue(columnFam, toBytes("rootId"));
        System.out.println(toInt(rootId));

        byte[] status = result.getValue(columnFam, toBytes("status"));
        System.out.println(toInt(status));
      }
    }

  }
}
