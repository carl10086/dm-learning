package com.ysz.dm.netty.dm.basic.recycler;

import io.netty.util.Recycler;
import io.netty.util.Recycler.Handle;
import lombok.Getter;
import lombok.ToString;

public class RecyclerDm {

  public static void main(String[] args) {
    Recycler<TargetObj> recycler = new Recycler<TargetObj>() {
      @Override
      protected TargetObj newObject(final Handle<TargetObj> handle) {
        System.err.println("newObject");
        TargetObj targetObj = new TargetObj();
        targetObj.handle = handle;
        return targetObj;
      }
    };

    System.err.println("1: 第一次构建会调用构造器");
    TargetObj targetObj = recycler.get();
    System.err.println("2: 虽然构建，但是没有加入池对象中");
    recycler.get();
    targetObj.handle.recycle(targetObj);
    System.err.println("3: 不会触发构建，因为已经进入了 池中");
    recycler.get();
    System.err.println("4: 会构建、因为池中已经取出来了");
    recycler.get();
  }

  @ToString
  @Getter
  private static class TargetObj {

    private Handle<TargetObj> handle;
    private Long userId = 1L;
    private String info = "ok";


  }

}
