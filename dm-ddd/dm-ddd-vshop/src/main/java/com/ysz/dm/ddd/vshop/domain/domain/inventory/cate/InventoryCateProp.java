package com.ysz.dm.ddd.vshop.domain.domain.inventory.cate;

import com.ysz.dm.ddd.vshop.domain.domain.common.BaseEntity;
import java.util.List;
import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import lombok.ToString;
import org.apache.logging.log4j.util.Strings;

/**
 * @author carl
 * @create 2022-10-24 5:29 PM
 **/
@ToString
@Getter
@Setter
public final class InventoryCateProp extends BaseEntity<InventoryCatePropId> {

  @NonNull
  private final InventoryCatePropId id;

  @NonNull
  private final String name;

  private String desc = Strings.EMPTY;

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


  public InventoryCateProp(@NonNull Long id, @NonNull String name) {
    this.id = new InventoryCatePropId(id);
    this.name = name;
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
