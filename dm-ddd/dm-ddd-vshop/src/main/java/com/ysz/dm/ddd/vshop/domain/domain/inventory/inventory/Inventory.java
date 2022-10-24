package com.ysz.dm.ddd.vshop.domain.domain.inventory.inventory;

import com.ysz.dm.ddd.vshop.domain.domain.common.BaseEntity;
import com.ysz.dm.ddd.vshop.domain.domain.inventory.cate.InventoryCateId;
import com.ysz.dm.ddd.vshop.domain.domain.inventory.cate.InventoryCatePropId;
import java.util.Map;
import lombok.Getter;
import lombok.ToString;

/**
 * @author carl
 * @create 2022-10-24 6:16 PM
 **/
@ToString
@Getter
public class Inventory extends BaseEntity {

  private InventoryId id;

  private InventoryName name;

  private InventorySaleTime saleTime;

  private InventoryCateId cateId;

  /**
   * props maybe unstable
   */
  private Map<InventoryCatePropId, String> props;

  private Map<InventoryAttrKey, InventoryAttr> attrs;

  public Inventory setId(InventoryId id) {
    this.id = id;
    return this;
  }

  public Inventory setCateId(InventoryCateId cateId) {
    this.cateId = cateId;
    return this;
  }

  public Inventory setProps(Map<InventoryCatePropId, String> props) {
    this.props = props;
    return this;
  }

  public Inventory setName(InventoryName name) {
    this.name = name;
    return this;
  }

  public Inventory setSaleTime(InventorySaleTime saleTime) {
    this.saleTime = saleTime;
    return this;
  }
}
