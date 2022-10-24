package com.ysz.dm.ddd.vshop.domain.domain.inventory.cate;

import com.ysz.dm.ddd.vshop.domain.domain.common.BaseEntity;
import java.util.List;
import lombok.Getter;
import lombok.ToString;

/**
 * @author carl
 * @create 2022-10-24 5:29 PM
 **/
@ToString
@Getter
public final class InventoryCateProp extends BaseEntity {

  private final InventoryCatePropId id;

  /***
   * key name
   */
  private final String name;

  /**
   * optional , this inventory is needed ?
   */
  private boolean optional = false;

  /**
   * is salable property ? if true used as sku prop
   */
  private boolean sku = false;

  /**
   * type ..
   */
  private InventoryCatePropType type = InventoryCatePropType.selected;

  /**
   * selected values
   */
  private List<String> selectedValues;


  public InventoryCateProp(String name, InventoryCatePropId id) {
    this.name = name;
    this.id = id;
  }

  public InventoryCateProp setOptional(boolean optional) {
    this.optional = optional;
    return this;
  }

  public InventoryCateProp setSku(boolean sku) {
    this.sku = sku;
    return this;
  }


  public InventoryCateProp setType(InventoryCatePropType type) {
    this.type = type;
    return this;
  }

  public InventoryCateProp setSelectedValues(List<String> selectedValues) {
    this.selectedValues = selectedValues;
    return this;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }

    InventoryCateProp that = (InventoryCateProp) o;

    return id.equals(that.id);
  }

  @Override
  public int hashCode() {
    return id.hashCode();
  }
}
