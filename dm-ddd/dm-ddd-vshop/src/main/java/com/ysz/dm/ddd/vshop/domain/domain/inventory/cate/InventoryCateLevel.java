package com.ysz.dm.ddd.vshop.domain.domain.inventory.cate;

/**
 * @author carl
 * @create 2022-10-27 11:09 AM
 **/
public enum InventoryCateLevel {
  one(1),
  two(2),
  three(3);


  private final int val;

  InventoryCateLevel(int val) {
    this.val = val;
  }
}
