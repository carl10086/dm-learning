package com.ysz.dm.ddd.vshop.domain.domain.inventory.cate;

import com.ysz.dm.ddd.vshop.domain.domain.common.BaseEntity;
import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

/**
 * inventory cate key make decision of a collection of inventory properties
 *
 * @author carl
 * @create 2022-10-24 5:25 PM
 **/
@Getter
@ToString
@Setter
public final class InventoryCate extends BaseEntity<InventoryCateId> {

  private final InventoryCateId id;

  /*all ids to represent full inherited relationships*/
  private final InventoryCateIds ids;

  /*all props*/
  private InventoryCateProps props;

  public InventoryCate(InventoryCateIds ids) {
    this.ids = ids;
    this.id = ids.currentId();
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }

    InventoryCate that = (InventoryCate) o;

    return id.equals(that.id);
  }

  @Override
  public int hashCode() {
    return id.hashCode();
  }
}
