package com.ysz.dm.ddd.vshop.domain.domain.subscribe;

import com.ysz.dm.ddd.vshop.domain.domain.common.BaseEntity;
import com.ysz.dm.ddd.vshop.domain.domain.payment.PaymentChannel;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.ToString;

/**
 * <pre>
 * 一次用户的签约
 * </pre>
 *
 * @author carl.yu
 * @createAt 2022/11/9
 **/
@ToString
@Getter
@RequiredArgsConstructor
public class SubSign extends BaseEntity<SubSignId> {

  private final SubSignId id;

  private final PaymentChannel paymentChannel;

  private SubSignDetail signDetail;


}
