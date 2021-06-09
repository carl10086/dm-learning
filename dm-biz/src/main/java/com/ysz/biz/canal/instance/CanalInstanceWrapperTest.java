package com.ysz.biz.canal.instance;

import com.alibaba.otter.canal.instance.manager.CanalInstanceWithManager;
import com.alibaba.otter.canal.instance.manager.model.Canal;
import com.alibaba.otter.canal.protocol.CanalEntry.Column;
import com.alibaba.otter.canal.protocol.CanalEntry.Entry;
import com.alibaba.otter.canal.protocol.CanalEntry.EntryType;
import com.alibaba.otter.canal.protocol.CanalEntry.EventType;
import com.alibaba.otter.canal.protocol.CanalEntry.Header;
import com.alibaba.otter.canal.protocol.CanalEntry.RowChange;
import com.alibaba.otter.canal.protocol.CanalEntry.RowData;
import com.alibaba.otter.canal.protocol.ClientIdentity;
import com.alibaba.otter.canal.protocol.Message;
import com.alibaba.otter.canal.server.embedded.CanalServerWithEmbedded;
import java.util.List;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

public class CanalInstanceWrapperTest {


  private CanalServerWithEmbedded server;
  private String tst = "example";
  //  protected static final String FILTER = ".*\\\\..*";
//  protected static final String FILTER = null;
  protected static final String FILTER = "zcwdb\\.auth_user";

  @Before
  public void setUp() throws Exception {
    this.server = CanalServerWithEmbedded.instance();
    this.server.setCanalInstanceGenerator(destination -> {
      Canal canal = CanalInstanceWrapper.buildCanal();
      return new CanalInstanceWithManager(canal, FILTER);
    });

    this.server.start();
    this.server.start(tst);
    System.err.println("started");
  }

  @Test
  public void tst() throws Exception {
    ClientIdentity clientIdentity = new ClientIdentity(tst, (short) 1);
    this.server.subscribe(clientIdentity);
    while (true) {
      Message message = this.server.getWithoutAck(clientIdentity, 100);
      List<Entry> entries = message.getEntries();
      if (entries != null && entries.size() > 0) {
        printEntry(entries);
        System.err.println("get successeed");
      } else {
        Thread.sleep(1000L);
      }
    }
  }

  private static void printEntry(List<Entry> entrys) {
    for (Entry entry : entrys) {
      if (entry.getEntryType() == EntryType.TRANSACTIONBEGIN
          || entry.getEntryType() == EntryType.TRANSACTIONEND) {
        continue;
      }

      RowChange rowChage = null;
      try {
        rowChage = RowChange.parseFrom(entry.getStoreValue());
      } catch (Exception e) {
        throw new RuntimeException(
            "ERROR ## parser of eromanga-event has an error , data:" + entry.toString(),
            e);
      }

      EventType eventType = rowChage.getEventType();
      Header header = entry.getHeader();
      System.out.println(
          String.format("================&gt; binlog[%s:%s] , name[%s,%s] , eventType : %s",
              header.getLogfileName(), header.getLogfileOffset(),
              header.getSchemaName(), header.getTableName(),
              eventType));

      for (RowData rowData : rowChage.getRowDatasList()) {
        if (eventType == EventType.DELETE) {
          printColumn(rowData.getBeforeColumnsList());
        } else if (eventType == EventType.INSERT) {
          printColumn(rowData.getAfterColumnsList());
        } else {
          System.out.println("-------&gt; before");
          printColumn(rowData.getBeforeColumnsList());
          System.out.println("-------&gt; after");
          printColumn(rowData.getAfterColumnsList());
        }
      }
    }
  }

  private static void printColumn(List<Column> columns) {
    for (Column column : columns) {
      System.out.println(
          column.getName() + " : " + column.getValue() + "    update=" + column.getUpdated());
    }
  }

  @After
  public void destory() {
    this.server.stop();
  }
}
