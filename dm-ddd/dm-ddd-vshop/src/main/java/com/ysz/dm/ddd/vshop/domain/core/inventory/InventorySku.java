package com.ysz.dm.ddd.vshop.domain.core.inventory;

import com.ysz.dm.ddd.vshop.domain.core.common.sku.SkuId;
import com.ysz.dm.ddd.vshop.domain.core.inventory.cate.InventoryCateProperties;
import lombok.Getter;
import lombok.ToString;

/**
 * @author carl
 * @create 2022-09-09 2:34 PM
 **/
@ToString
@Getter
public class InventorySku {

  /**
   * sku id
   */
  private final SkuId id;

  private InventoryCateProperties inventoryCateProperties;

  private InventoryPrice price;


  public InventorySku(SkuId id) {
    this.id = id;
  }

  public InventorySku setInventoryCateProperties(InventoryCateProperties inventoryCateProperties) {
    this.inventoryCateProperties = inventoryCateProperties;
    return this;
  }

  public InventorySku setPrice(InventoryPrice price) {
    this.price = price;
    return this;
  }
}
