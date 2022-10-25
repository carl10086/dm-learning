package com.ysz.dm.ddd.vshop.domain.domain.inventory.attrs;

import lombok.Getter;
import lombok.ToString;

/**
 * @author carl
 * @create 2022-10-24 6:32 PM
 **/
@ToString
@Getter
public class InventoryAttr {

  private InventoryAttrKey key;
  private InventoryAttrValueType type;
  private Object value;

  public InventoryAttr setKey(InventoryAttrKey key) {
    this.key = key;
    return this;
  }

  public InventoryAttr setType(InventoryAttrValueType type) {
    this.type = type;
    return this;
  }

  public InventoryAttr setValue(Object value) {
    this.value = value;
    return this;
  }
}
