package com.ysz.dm.base.repo.support.mapping

import org.springframework.data.util.TypeInformation

/**
 * @author carl
 * @since 2023-02-19 3:40 PM
 **/
class PropertyPath(
    val name: String,
    val owningType: TypeInformation<*>
) {
}