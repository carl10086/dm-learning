package com.ysz.biz.mysql.binlog;

import com.github.shyiko.mysql.binlog.BinaryLogFileReader;
import com.github.shyiko.mysql.binlog.event.DeleteRowsEventData;
import com.github.shyiko.mysql.binlog.event.Event;
import com.github.shyiko.mysql.binlog.event.EventType;
import com.github.shyiko.mysql.binlog.event.TableMapEventData;
import com.github.shyiko.mysql.binlog.event.UpdateRowsEventData;
import com.github.shyiko.mysql.binlog.event.WriteRowsEventData;
import com.github.shyiko.mysql.binlog.event.deserialization.EventDeserializer;
import java.io.File;
import java.io.Serializable;
import java.util.BitSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import org.apache.commons.lang3.time.FastDateFormat;

public class MysqlBinlogParser_Dm_001 {

  public static void main(String[] args) throws Exception {
//    File binlogFile = new File("/Users/carl/tmp/useless/101-bin.002155");
    File binlogFile = new File("/Users/carl/tmp/useless/33-bin.000001");
    EventDeserializer eventDeserializer = new EventDeserializer();
    eventDeserializer.setCompatibilityMode(
        EventDeserializer.CompatibilityMode.DATE_AND_TIME_AS_LONG,
        EventDeserializer.CompatibilityMode.CHAR_AND_BINARY_AS_BYTE_ARRAY
    );
    BinaryLogFileReader reader = new BinaryLogFileReader(binlogFile, eventDeserializer);
    Map<EventType, Event> datumMap = new HashMap<>();
    String tableName = "";
    Event prevTableMapEvent = null;
    try {
      for (Event event; (event = reader.readEvent()) != null; ) {
        String date = FastDateFormat.getInstance("yyyyMMdd HH:mm:ss")
            .format(event.getHeader().getTimestamp());
        EventType eventType = event.getHeader().getEventType();
        if (eventType == EventType.TABLE_MAP) {
          TableMapEventData tableMapEventData = event.getData();
          tableName = tableMapEventData.getTable();
          prevTableMapEvent = event;
        }
        final String debugTableName = "tst_binlog";
//        final String debugTableName = "message_message";

        if (tableName.equalsIgnoreCase(debugTableName) && EventType.isWrite(eventType)) {
          System.out.println("insert");
          WriteRowsEventData writeRowsEventData = event.getData();
          List<Serializable[]> rows = writeRowsEventData.getRows();
          for (Serializable[] row : rows) {
            /*插入的值*/
            System.out.println(row);
          }
        }

        if (tableName.equalsIgnoreCase(debugTableName) && EventType.isUpdate(eventType)) {
          UpdateRowsEventData updateRowsEventData = event.getData();
          BitSet includedColumnsBeforeUpdate = updateRowsEventData.getIncludedColumnsBeforeUpdate();
          List<Entry<Serializable[], Serializable[]>> rows = updateRowsEventData.getRows();
          for (Entry<Serializable[], Serializable[]> row : rows) {
            /*key 应该是修改之前的值*/
            Serializable[] key = row.getKey();
            /*key 是修改之后的值*/
            Serializable[] value = row.getValue();
            System.err.println("finish");
          }
        }

        if (tableName.equalsIgnoreCase(debugTableName) && EventType.isDelete(eventType)) {
          DeleteRowsEventData data = event.getData();
          List<Serializable[]> rows = data.getRows();
          for (Serializable[] row : rows) {
            System.err.println(row);
          }
        }

        if (!datumMap.containsKey(eventType)) {
          datumMap.put(eventType, event);
        }
      }
    } finally {
      reader.close();
    }

    System.err.println(1);
  }

}
