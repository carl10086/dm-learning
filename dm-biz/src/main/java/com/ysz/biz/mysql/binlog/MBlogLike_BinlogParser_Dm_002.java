package com.ysz.biz.mysql.binlog;

import com.github.shyiko.mysql.binlog.BinaryLogFileReader;
import com.github.shyiko.mysql.binlog.event.Event;
import com.github.shyiko.mysql.binlog.event.EventData;
import com.github.shyiko.mysql.binlog.event.EventHeaderV4;
import com.github.shyiko.mysql.binlog.event.EventType;
import com.github.shyiko.mysql.binlog.event.TableMapEventData;
import com.github.shyiko.mysql.binlog.event.WriteRowsEventData;
import com.github.shyiko.mysql.binlog.event.deserialization.EventDeserializer;
import java.io.File;
import java.io.Serializable;
import java.nio.charset.Charset;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.time.FastDateFormat;

public class MBlogLike_BinlogParser_Dm_002 {

  private static final long ONE_HOUR = 1L * 3600L * 1000L;

  public static void main(String[] args) throws Exception {
    File binlogFile = new File("/Users/carl/tmp/useless/s50-bin.061787");
    EventDeserializer eventDeserializer = new EventDeserializer();
    eventDeserializer.setCompatibilityMode(
        EventDeserializer.CompatibilityMode.DATE_AND_TIME_AS_LONG,
        EventDeserializer.CompatibilityMode.CHAR_AND_BINARY_AS_BYTE_ARRAY
    );
    BinaryLogFileReader reader = new BinaryLogFileReader(binlogFile, eventDeserializer);
    Map<EventType, Event> datumMap = new HashMap<>();
    final String debugTableName = "mblog_like";
    String tableName = "";
    Event prevTableMapEvent = null;
    try {
      for (Event event; (event = reader.readEvent()) != null; ) {
        final EventHeaderV4 header = event.getHeader();
        long nextPosition = header.getNextPosition();
//        if (nextPosition == 86726560L) {
//          System.err.println("事件开始");
//        } else {
//          continue;
//        }
        final EventData data = event.getData();
        final String timeStr = FastDateFormat.getInstance("yyyyMMdd HH:mm:ss")
            .format(header.getTimestamp());
        final EventType eventType = event.getHeader().getEventType();
        if (eventType == EventType.TABLE_MAP) {
          TableMapEventData tableMapEventData = event.getData();
          tableName = tableMapEventData.getTable();
          prevTableMapEvent = event;
        }

        File file = new File("/Users/carl/tmp/useless/like.txt");
        if (debugTableName.equalsIgnoreCase(tableName) && EventType.isWrite(eventType)) {
          System.err.println("find insert");
          WriteRowsEventData writeRowsEventData = event.getData();
          List<Serializable[]> rows = writeRowsEventData.getRows();
          for (Serializable[] row : rows) {
            System.err.println(row);
            Integer senderId = (Integer) row[1];
            Integer blogId = (Integer) row[2];
            Long createDate = (Long) row[3] - ONE_HOUR * 8;
            String date = FastDateFormat.getInstance("yyyy-MM-dd HH:mm:ss").format(createDate);
            Integer category = (Integer) row[4];
            if (category != 1) {
              continue;
            }

            MBlogLikeEvent likeEvent = new MBlogLikeEvent(
                senderId.longValue(),
                true,
                blogId.longValue(),
                date
            );
            FileUtils.writeStringToFile(file, likeEvent + "\n", Charset.defaultCharset(), true);
          }

        }
      }
    } finally {
      reader.close();
    }
  }

}
