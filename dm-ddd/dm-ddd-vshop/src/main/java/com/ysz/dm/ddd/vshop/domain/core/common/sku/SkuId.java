package com.ysz.dm.ddd.vshop.domain.core.common.sku;

import lombok.Getter;
import lombok.ToString;

/**
 * @author carl
 * @create 2022-09-09 2:32 PM
 **/
@ToString
@Getter
public class SkuId {

  private final Long id;

  public SkuId(Long id) {
    this.id = id;
  }
}
