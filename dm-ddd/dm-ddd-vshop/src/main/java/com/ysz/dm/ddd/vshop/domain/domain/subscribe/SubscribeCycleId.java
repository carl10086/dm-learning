package com.ysz.dm.ddd.vshop.domain.domain.subscribe;

import com.ysz.dm.ddd.vshop.domain.domain.payment.PaymentChannel;

/**
 * @author carl
 * @create 2022-11-10 11:14 AM
 **/
public record SubscribeCycleId(
    String thirdSignId,
    PaymentChannel channel,
    String uuid
) {

}
