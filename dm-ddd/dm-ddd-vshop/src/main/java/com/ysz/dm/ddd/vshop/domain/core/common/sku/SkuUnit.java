package com.ysz.dm.ddd.vshop.domain.core.common.sku;

import lombok.Getter;
import lombok.ToString;

/**
 * @author carl
 * @create 2022-09-09 2:36 PM
 **/
@ToString
@Getter
public class SkuUnit {

  private final String unit;

  public SkuUnit(String unit) {
    this.unit = unit;
  }
}
