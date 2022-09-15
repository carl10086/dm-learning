package com.ysz.dm.ddd.vshop.domain.core.inventory.cate;

import java.util.List;
import lombok.Getter;
import lombok.ToString;
import org.apache.commons.lang3.builder.EqualsBuilder;
import org.apache.commons.lang3.builder.HashCodeBuilder;

/**
 * @author carl
 * @create 2022-09-14 12:26 PM
 **/
@ToString
@Getter
public class InventoryCatePropertyKey<V> {

  private InventoryCatePropertyKeyId id;

  private Boolean saleable = true;

  private String name;

  private InventoryCatePropertyType type;

  private Class<V> valueClass;

  private List<V> selectedValues;

  private String desc;


  public InventoryCatePropertyKey<V> setId(InventoryCatePropertyKeyId id) {
    this.id = id;
    return this;
  }

  public InventoryCatePropertyKey<V> setName(String name) {
    this.name = name;
    return this;
  }

  public InventoryCatePropertyKey<V> setType(InventoryCatePropertyType type) {
    this.type = type;
    return this;
  }

  public InventoryCatePropertyKey<V> setValueClass(Class<V> valueClass) {
    this.valueClass = valueClass;
    return this;
  }

  public InventoryCatePropertyKey<V> setSelectedValues(List<V> selectedValues) {
    this.selectedValues = selectedValues;
    return this;
  }

  public InventoryCatePropertyKey<V> setDesc(String desc) {
    this.desc = desc;
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

    InventoryCatePropertyKey<?> that = (InventoryCatePropertyKey<?>) o;

    return new EqualsBuilder().append(id, that.id).isEquals();
  }

  @Override
  public int hashCode() {
    return new HashCodeBuilder(17, 37).append(id).toHashCode();
  }
}
