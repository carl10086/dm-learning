package com.ysz.dm.ddd.vshop.domain.core.inventory.cate;

import java.util.SortedSet;
import lombok.Getter;
import lombok.ToString;

/**
 * 商品类目
 *
 * @author carl
 * @create 2022-09-14 12:07 PM
 **/
@ToString
@Getter
public class InventoryCate {

  private final InventoryCateId cateId;
  private SortedSet<InventoryCatePropertyKey> keys;


  public InventoryCate(InventoryCateId cateId) {
    this.cateId = cateId;
  }

  public InventoryCate setKeys(SortedSet<InventoryCatePropertyKey> keys) {
    this.keys = keys;
    return this;
  }
}
