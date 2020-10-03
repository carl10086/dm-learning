package com.ysz.biz.mysql.binlog;

import ch.qos.logback.core.boolex.EventEvaluator;
import com.github.shyiko.mysql.binlog.BinaryLogFileReader;
import com.github.shyiko.mysql.binlog.event.DeleteRowsEventData;
import com.github.shyiko.mysql.binlog.event.Event;
import com.github.shyiko.mysql.binlog.event.EventData;
import com.github.shyiko.mysql.binlog.event.EventHeaderV4;
import com.github.shyiko.mysql.binlog.event.EventType;
import com.github.shyiko.mysql.binlog.event.TableMapEventData;
import com.github.shyiko.mysql.binlog.event.UpdateRowsEventData;
import com.github.shyiko.mysql.binlog.event.WriteRowsEventData;
import com.github.shyiko.mysql.binlog.event.deserialization.EventDeserializer;
import java.io.File;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Map.Entry;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.time.FastDateFormat;

public class MessageMessageBinlogParser {

  private final String debugTableName = "message_message";

  private final Long debugPos = 18530606L;

  private final EventBuffer eventBuffer = new EventBuffer(5);

  private static final long ONE_HOUR = 1L * 3600L * 1000L;


  private void parse() throws Exception {
    File binlogFile = new File("/Users/carl.yu/tmp/useless/101-bin.002190");
    EventDeserializer eventDeserializer = new EventDeserializer();
    eventDeserializer.setCompatibilityMode(
        EventDeserializer.CompatibilityMode.DATE_AND_TIME_AS_LONG,
        EventDeserializer.CompatibilityMode.CHAR_AND_BINARY_AS_BYTE_ARRAY
    );
    BinaryLogFileReader reader = new BinaryLogFileReader(binlogFile, eventDeserializer);

    Event prevTableMapEvent = null;
    String tableName = "";
//    EventBuffer<MessageFavEvent> messageFavEventEventBuffer = new EventBuffer<>(50);
    List<String> messageEvent = new ArrayList<>();
    try {
      for (Event event; (event = reader.readEvent()) != null; ) {
        eventBuffer.add(event);
        final EventHeaderV4 header = event.getHeader();
        long nextPosition = header.getNextPosition();
        final EventData data = event.getData();
        final String timeStr = FastDateFormat.getInstance("yyyyMMdd HH:mm:ss")
            .format(header.getTimestamp());
        final EventType eventType = event.getHeader().getEventType();
        if (eventType == EventType.TABLE_MAP) {
          /*1. 记录时间状态*/
          TableMapEventData tableMapEventData = event.getData();
          tableName = tableMapEventData.getTable();
          prevTableMapEvent = event;
        }

        if (nextPosition == debugPos) {
          messageEvent.add("breakPoint:" + debugPos);
          System.err.println("开始");
        }

        if (nextPosition == debugPos) {
          System.err.println("开始计数");
        }

        if (debugTableName.equalsIgnoreCase(tableName) && EventType.isWrite(eventType)) {
          System.err.println("find insert");
          WriteRowsEventData writeRowsEventData = event.getData();
          List<Serializable[]> rows = writeRowsEventData.getRows();
          for (Serializable[] row : rows) {
            System.err.println(row);
            Integer afterStatus = (Integer) row[6];
            Integer id = (Integer) row[0];
            Long addTime = (Long) row[5] - ONE_HOUR * 8;
            messageEvent.add(new MessageFavEvent(
                id.longValue(), FastDateFormat.getInstance("yyyyMMdd HH:mm:ss").format(addTime),
                "insert", nextPosition
            ) + "");
          }
        }

        if (debugTableName.equalsIgnoreCase(tableName) && EventType.isUpdate(eventType)) {
          UpdateRowsEventData updateRowsEventData = event.getData();
          List<Entry<Serializable[], Serializable[]>> rows = updateRowsEventData.getRows();
          for (Entry<Serializable[], Serializable[]> row : rows) {
            Integer prevStatus = (Integer) row.getKey()[6];
            Integer afterStatus = (Integer) row.getValue()[6];
            Integer id = (Integer) row.getValue()[0];
            Long addTime = (Long) row.getValue()[5] - ONE_HOUR * 8;
            if (afterStatus == 5) {
              messageEvent.add(new MessageFavEvent(
                  id.longValue(), FastDateFormat.getInstance("yyyyMMdd HH:mm:ss").format(addTime),
                  "delete", nextPosition
              ) + "");
            }
          }
          System.err.println("delete");
        }
      }
    } finally {
      reader.close();
    }

    FileUtils.writeLines(new File("/Users/carl.yu/tmp/useless/result.txt"), messageEvent);

  }

  public static void main(String[] args) throws Exception {
    new MessageMessageBinlogParser().parse();
  }

}
