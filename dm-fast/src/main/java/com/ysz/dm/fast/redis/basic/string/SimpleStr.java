package com.ysz.dm.fast.redis.basic.string;

public interface SimpleStr {


  SimpleStrCate cate();

  /**
   * char 数组 分配时候的总长度 ..
   * @return
   */
  int total();

  /**
   * 已经使用的长度
   * @return
   */
  int used();
}
