package com.ysz.dm.ddd.vshop.domain.domain.inventory.inventory;

import com.ysz.dm.ddd.vshop.domain.domain.common.BaseEntity;
import com.ysz.dm.ddd.vshop.domain.domain.inventory.attrs.InventoryAttrs;
import com.ysz.dm.ddd.vshop.domain.domain.inventory.cate.InventoryCateId;
import com.ysz.dm.ddd.vshop.domain.domain.inventory.cate.InventoryCateProps;
import lombok.Getter;
import lombok.ToString;

/**
 * @author carl
 * @create 2022-10-24 6:16 PM
 **/
@ToString
@Getter
public class Inventory extends BaseEntity<InventoryId> {

  private final InventoryId id;

  private InventoryName name;

  private InventorySaleTime saleTime;

  private InventoryCateId cateId;

  /**
   * dynamic props decided by inventory category
   */
  private InventoryCateProps cateProps;

  private InventoryAttrs attrs;


  public Inventory(InventoryId id) {
    this.id = id;
  }
}
