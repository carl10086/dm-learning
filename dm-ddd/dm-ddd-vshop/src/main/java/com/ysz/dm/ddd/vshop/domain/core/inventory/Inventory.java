package com.ysz.dm.ddd.vshop.domain.core.inventory;

import lombok.Getter;
import lombok.ToString;

/**
 * @author carl
 * @create 2022-09-09 10:23 AM
 **/
@ToString
@Getter
public class Inventory {

  /**
   * 商品 id
   */
  private InventoryId id;

  /**
   * 标题
   */
  private InventoryTitle title;



}
