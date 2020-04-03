package com.ysz.dm.fast.kryo;

import cn.hutool.core.util.IdUtil;
import com.alibaba.fastjson.JSON;
import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Output;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.List;
import org.junit.Test;
import org.nustaq.serialization.FSTConfiguration;
import org.nustaq.serialization.FSTObjectOutput;

/**
 * @author carl.yu
 * @date 2020/3/17
 */
public class KryoDm {

  static FSTConfiguration conf = FSTConfiguration.createDefaultConfiguration();

  public int jsonBytes(ComplexObj obj) throws IOException {
    String s = JSON.toJSONString(obj);
    return s.getBytes().length;
  }

  public int javaBytes(ComplexObj obj) throws IOException {
    ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
    ObjectOutputStream stream = new ObjectOutputStream(outputStream);
    stream.writeObject(obj);
    return outputStream.toByteArray().length;
  }

  public int fastSerialization(Object obj) throws IOException {
    ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
    FSTObjectOutput out = conf.getObjectOutput(outputStream);
    out.writeObject(obj, obj.getClass());
    out.flush();
    outputStream.close();
    return outputStream.toByteArray().length;
  }

  public int kryoWithRegisty(Object obj) {
    Kryo kryo = new Kryo();
    kryo.register(ComplexObj.class);
    kryo.register(ArrayList.class);
    ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
    Output output = new Output(outputStream);
    kryo.writeObject(output, obj);
    output.close();
    return outputStream.toByteArray().length;
  }

  @Test
  public void tstBytes() throws IOException {
    ComplexObj obj = mockComplexObj();
    System.out.println("java:" + javaBytes(obj));
    System.out.println("json:" + jsonBytes(obj));
    System.out.println("fst: " + fastSerialization(obj));
    System.out.println("kryo:" + kryoWithRegisty(obj));
    List<ComplexObj> complexObjs = obj.asPlainObjs();
    System.out.println("kryo with plain:" + kryoWithRegisty(complexObjs));
    System.out.println("fst with plain:" + fastSerialization(complexObjs));

  }

  private SimpleObj mockCustomObj() {
    SimpleObj customObj = new SimpleObj();
    customObj.setStartAt((int) (System.currentTimeMillis() / 1000L));
    customObj.setDuration(5);
    customObj.setApp("saturn");
    customObj.setName("aaa.bbb.ccc");
    customObj.setType("repo");
    return customObj;
  }

  public ComplexObj mockComplexObj() {
    /*模拟5层递归*/
    ComplexObj obj = mockWithLevel(1, "parent");
    obj.setRootId(obj.getId());
    obj.setChildren(new ArrayList<>());
    for (int i = 0; i < 5; i++) {
      ComplexObj child = mockWithLevel(2, "-demo-" + i);
      obj.addChild(child);
      child.setChildren(new ArrayList<>());
      for (int j = 0; j < 5; j++) {
        ComplexObj third = mockWithLevel(3, "-demo-" + j);
        child.addChild(third);
      }
    }
    return obj;
  }


  public ComplexObj mockWithLevel(int level, String name) {
    ComplexObj complexObj = new ComplexObj();
    complexObj.setId(IdUtil.objectId());
    complexObj.setStartAt((int) (System.currentTimeMillis() / 1000L));
    complexObj.setDuration(2);
    complexObj.setApp("saturn");
    complexObj.setName("level-" + level + "-" + name);
    complexObj.setType("repo");
    return complexObj;
  }


}
