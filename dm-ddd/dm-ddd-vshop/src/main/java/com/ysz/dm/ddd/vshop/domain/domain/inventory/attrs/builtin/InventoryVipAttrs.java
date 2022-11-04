package com.ysz.dm.ddd.vshop.domain.domain.inventory.attrs.builtin;

/**
 * @author carl
 * @create 2022-10-27 10:42 AM
 **/
public record InventoryVipAttrs(
    /*true, 表明是 svip*/
    boolean svip,
    int months
) {

}
