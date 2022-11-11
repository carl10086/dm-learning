package com.ysz.dm.ddd.vshop.domain.domain.subscribe;

import com.ysz.dm.ddd.vshop.domain.domain.payment.PaymentChannel;

/**
 * @author carl
 * @create 2022-11-10 2:29 PM
 **/
public record SubscribeSignId(
    String thirdSignId,
    PaymentChannel channel
//    String uuid
) {

}
