package com.ysz.dm.ddd.vshop.domain.core.inventory.cate;

import com.ysz.dm.ddd.vshop.domain.core.common.lang.BaseLongId;
import lombok.Getter;
import lombok.ToString;

/**
 * @author carl
 * @create 2022-09-15 11:49 AM
 **/
@ToString
@Getter
public class InventoryCatePropertyKeyId extends BaseLongId {

  public InventoryCatePropertyKeyId(Long id) {
    super(id);
  }
}
