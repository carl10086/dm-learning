package com.ysz.dm.ddd.vshop.domain.core.inventory;

import lombok.Getter;
import lombok.ToString;

/**
 * 商品标题 可以是有 schema 的. 不同的时候展示逻辑不同 . .
 *
 * @author carl
 * @create 2022-09-09 10:26 AM
 **/
@ToString
@Getter
public class InventoryTitle {

  /**
   * 一级标题
   */
  private String levelOne;

}
