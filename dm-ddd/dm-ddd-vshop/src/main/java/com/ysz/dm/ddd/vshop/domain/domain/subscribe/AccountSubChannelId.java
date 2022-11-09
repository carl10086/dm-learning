package com.ysz.dm.ddd.vshop.domain.domain.subscribe;

import com.ysz.dm.ddd.vshop.domain.domain.distribution.domain.AccountId;
import com.ysz.dm.ddd.vshop.domain.domain.payment.PaymentChannel;

/**
 * <pre>
 * class desc here
 * </pre>
 *
 * @author carl.yu
 * @createAt 2022/11/9
 **/
public record AccountSubChannelId(
    AccountId accountId,
    PaymentChannel paymentChannel
) {

}
