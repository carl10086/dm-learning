package com.ysz.dm.ddd.vshop.domain.domain.subscribe;

import com.ysz.dm.ddd.vshop.domain.domain.acount.AccountId;
import com.ysz.dm.ddd.vshop.domain.domain.payment.PaymentChannel;

/**
 * @author carl
 * @create 2022-11-10 11:15 AM
 **/
public record SubscribeId(
    AccountId accountId,
    PaymentChannel paymentChannel
) {

}