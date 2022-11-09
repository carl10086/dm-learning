package com.ysz.dm.ddd.vshop.domain.domain.subscribe;

/**
 * <pre>
 * class desc here
 * </pre>
 *
 * @author carl.yu
 * @createAt 2022/11/9
 **/
public record SubSignId(
    /*堆糖内部生成的签约id*/
    String signId,
    /*外部系统提供的唯一签约id*/
    String thirdId) {

}
