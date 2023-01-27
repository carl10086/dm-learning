package com.ysz.dm.base.redis.ratelimit

import com.ysz.dm.base.core.validate.ApiParamValidateTools.Companion.chk
import com.ysz.dm.base.core.validate.Validatable
import java.time.Duration

/**
 * <pre>
 *     滑动窗口请求
 * </pre>
 * @author carl
 * @create 2022-12-08 3:56 PM
 **/
data class SlidingWindowReq(
    /*时间窗口大小*/
    val timeWindowUnit: Duration,
    val unitNum: Int = 2,
    /*单位时间窗口中最大的允许的量*/
    val maxReq: Int,
    /*毫秒时间戳*/
    val occurAt: Long,
    val key: String
) : Validatable {
    override fun validate() {
        chk(unitNum >= 2, "unit Num must >= 2, if = 1, use fixed window")
        chk(
            unitNum <= 30,
            "currently only support max 30 units for sliding window, consideration of algorithm performance"
        )
        chk(maxReq > 0, "max Req must > 1")
    }
}
