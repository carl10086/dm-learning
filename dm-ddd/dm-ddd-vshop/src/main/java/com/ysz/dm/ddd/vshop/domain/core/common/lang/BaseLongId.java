package com.ysz.dm.ddd.vshop.domain.core.common.lang;

import lombok.Getter;
import lombok.ToString;

/**
 * @author carl
 * @create 2022-09-15 4:16 PM
 **/
@ToString
@Getter
public abstract class BaseLongId {

  protected final Long id;

  public BaseLongId(Long id) {
    this.id = id;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }

    BaseLongId baseLongId = (BaseLongId) o;

    return id.equals(baseLongId.id);
  }

  @Override
  public int hashCode() {
    return id.hashCode();
  }
}
