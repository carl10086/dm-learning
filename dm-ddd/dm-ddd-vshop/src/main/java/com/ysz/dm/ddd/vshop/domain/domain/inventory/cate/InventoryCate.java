package com.ysz.dm.ddd.vshop.domain.domain.inventory.cate;

import com.ysz.dm.ddd.vshop.domain.domain.common.BaseEntity;
import java.util.List;
import lombok.Getter;
import lombok.ToString;

/**
 * inventory cate key make decision of a collection of inventory properties
 *
 * @author carl
 * @create 2022-10-24 5:25 PM
 **/
@Getter
@ToString
public final class InventoryCate extends BaseEntity {

  private final InventoryCateId id;
  private final InventoryCateId parentId;

  private List<InventoryCateProp> props;

  public InventoryCate(InventoryCateId id, InventoryCateId parentId) {
    this.id = id;
    this.parentId = parentId;
  }

  public InventoryCate setProps(List<InventoryCateProp> props) {
    this.props = props;
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

    InventoryCate that = (InventoryCate) o;

    return id.equals(that.id);
  }

  @Override
  public int hashCode() {
    return id.hashCode();
  }
}
