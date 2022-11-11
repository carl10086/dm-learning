package com.ysz.dm.ddd.vshop.domain.domain.subscribe;

import com.ysz.dm.ddd.vshop.domain.domain.common.BaseEntity;
import java.time.Instant;
import java.util.Date;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.ToString;

/**
 * 一次签约
 *
 * - 签约商品
 *
 * @author carl
 * @create 2022-11-10 11:12 AM
 **/
@ToString
@Getter
@RequiredArgsConstructor
public class SubscribeSign extends BaseEntity<SubscribeSignId> {

  private final Date signAt;

  private Instant renewAbleAt;


  @Override
  protected SubscribeSignId id() {
    return null;
  }
}
