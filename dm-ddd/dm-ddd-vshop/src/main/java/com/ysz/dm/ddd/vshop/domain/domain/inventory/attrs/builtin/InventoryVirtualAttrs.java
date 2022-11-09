package com.ysz.dm.ddd.vshop.domain.domain.inventory.attrs.builtin;

import com.ysz.dm.ddd.vshop.domain.domain.common.Rmb;
import com.ysz.dm.ddd.vshop.domain.domain.inventory.virtual.InventoryVirtualType;
import com.ysz.dm.ddd.vshop.domain.domain.payment.PaymentChannel;
import java.util.Set;

/**
 * @author carl
 * @create 2022-10-27 10:43 AM
 **/
public record InventoryVirtualAttrs(
    /*vip price*/
    Rmb vipPrice,
    /*svip price*/
    Rmb svipPrice,
    /*virtual inventory type*/
    InventoryVirtualType virtualType,
    /*payment channels*/
    Set<PaymentChannel> payments
) {

}
