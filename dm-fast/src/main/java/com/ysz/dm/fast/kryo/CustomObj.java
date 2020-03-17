package com.ysz.dm.fast.kryo;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import lombok.Data;

/**
 * @author carl.yu
 * @date 2020/3/17
 */
@Data
public class CustomObj implements Serializable {

  /**
   * 开始时间、秒
   */
  private int startAt;

  /**
   * 持续时间、秒
   */
  private int duration;

  /**
   * app 名称
   */
  private String app;

  /**
   * 事件名称
   */
  private String name;
  /**
   * 逻辑分组
   */
  private String type;

  public byte[] asBytes() {
    List<Byte> buffer = new ArrayList<>(1024);
    appendInt(buffer, startAt);
    appendInt(buffer, duration);
    appendString(buffer, app);
    appendString(buffer, name);
    appendString(buffer, type);

    byte[] res = new byte[buffer.size()];
    for (int i = 0; i < res.length; i++) {
      res[i] = buffer.get(i);
    }
    return res;
  }


  private void appendString(List<Byte> buffer, String data) {
    byte[] bytes = data.getBytes();
    buffer.add((byte) bytes.length);
    appendBytes(buffer, bytes);
  }

  private void appendBytes(List<Byte> buffer, byte[] data) {
    for (byte aByte : data) {
      buffer.add(aByte);
    }
  }

  private void appendInt(List<Byte> buffer, int data) {
    byte[] bytes = Bytes.int2bytes(data);
    appendBytes(buffer, bytes);
  }
}
