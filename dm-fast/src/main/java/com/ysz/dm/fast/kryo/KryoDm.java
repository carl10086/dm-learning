package com.ysz.dm.fast.kryo;

import com.alibaba.fastjson.JSON;
import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Output;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import org.junit.Test;

/**
 * @author carl.yu
 * @date 2020/3/17
 */
public class KryoDm {

  public int jsonBytes(CustomObj obj) throws IOException {
    String s = JSON.toJSONString(obj);
    return s.getBytes().length;
  }

  public int javaBytes(CustomObj obj) throws IOException {
    ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
    ObjectOutputStream stream = new ObjectOutputStream(outputStream);
    stream.writeObject(obj);
    return outputStream.toByteArray().length;
  }

  public int kryoWithRegisty(CustomObj obj) {
    Kryo kryo = new Kryo();
    kryo.register(CustomObj.class);
    ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
    Output output = new Output(outputStream);
    kryo.writeObject(output, obj);
    output.close();
    return outputStream.toByteArray().length;
  }

  @Test
  public void tstBytes() throws IOException {
    CustomObj obj = mockCustomObj();
    System.out.println("java:" + javaBytes(obj));
    System.out.println("json:" + jsonBytes(obj));
    System.out.println("kryo:" + kryoWithRegisty(obj));
    System.out.println("custom:" + obj.asBytes().length);
  }

  private CustomObj mockCustomObj() {
    CustomObj customObj = new CustomObj();
    customObj.setStartAt((int) (System.currentTimeMillis() / 1000L));
    customObj.setDuration(5);
    customObj.setApp("saturn");
    customObj.setName("aaa.bbb.ccc");
    customObj.setType("repo");
    return customObj;
  }


}
