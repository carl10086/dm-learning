package com.ysz.dm.ddd.vshop.domain.domain.inventory.attrs.builtin;

import com.ysz.dm.ddd.vshop.domain.domain.inventory.attrs.InventoryAttrs;
import java.util.List;
import java.util.Optional;
import lombok.Getter;
import lombok.ToString;

/**
 * @author carl
 * @create 2022-10-27 10:40 AM
 **/
@Getter
@ToString
public class InventoryMemoryAttrs implements InventoryAttrs {

  private InventoryVipAttrs vipAttrs;

  private InventoryVirtualAttrs virtualAttrs;

  @Override
  public Optional<InventoryVipAttrs> vip() {
    return Optional.of(vipAttrs);
  }

  @Override
  public Optional<InventoryVirtualAttrs> virtual() {
    return Optional.of(virtualAttrs);
  }

  public static InventoryAttrs merge(List<InventoryMemoryAttrs> memoryAttrsList) {
  }
}
