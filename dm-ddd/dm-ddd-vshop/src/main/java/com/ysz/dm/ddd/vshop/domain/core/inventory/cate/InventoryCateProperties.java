package com.ysz.dm.ddd.vshop.domain.core.inventory.cate;

import java.util.Map;
import lombok.Getter;
import lombok.ToString;

/**
 * @author carl
 * @create 2022-09-15 3:49 PM
 **/
@ToString
@Getter
public class InventoryCateProperties {


  private final Map<InventoryCatePropertyKeyId, Object> sortedMap;

  public InventoryCateProperties(Map<InventoryCatePropertyKeyId, Object> sortedMap) {
    this.sortedMap = sortedMap;
  }
}
