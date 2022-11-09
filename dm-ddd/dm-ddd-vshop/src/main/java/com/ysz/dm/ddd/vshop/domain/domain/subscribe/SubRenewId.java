package com.ysz.dm.ddd.vshop.domain.domain.subscribe;

import com.ysz.dm.ddd.vshop.domain.domain.payment.PaymentChannel;

/**
 * <pre>
 * 如何约束一次续订 id
 * </pre>
 *
 * @author carl.yu
 * @createAt 2022/11/9
 **/
public record SubRenewId(
    String uuid,
    String thirdRenewId,
    PaymentChannel channel
    ) {

}
