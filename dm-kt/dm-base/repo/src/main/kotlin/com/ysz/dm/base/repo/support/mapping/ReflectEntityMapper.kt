package com.ysz.dm.base.repo.support.mapping

import com.ysz.dm.base.repo.anotation.GeneratedValue
import com.ysz.dm.base.repo.anotation.Id
import com.ysz.dm.base.repo.anotation.Ignore
import java.sql.ResultSet
import kotlin.reflect.*
import kotlin.reflect.full.memberProperties
import kotlin.reflect.jvm.javaField
import kotlin.reflect.jvm.jvmErasure

/**
 * @author carl
 * @since 2023-02-18 1:48 AM
 **/
interface ReflectEntityMapper<T> {
    /**
     * return all columns as the property order of the domain class
     */
    fun columns(): List<String>

    /**
     * return all the values by the order of the domain class
     */
    fun allPropertyValues(entity: T, withPrimaryKey: Boolean = true): List<Any?>

    /**
     * a rowMapper used for query
     */
    fun rowMapper(): (ResultSet) -> T

    /**
     *  this support auto generate id by database . if True : yes
     */
    fun autoGenerateId(): Boolean

    /**
     * the property value of primary key
     */
    fun primaryKeyValue(entity: T): Any?

    fun setPrimaryKeyValue(entity: T, pkValue: Number)

    fun primaryKeyColumnName(): String
}

class KotlinDataClassMapper<T : Any>(
    private val kClass: KClass<T>,
    private val converter: PropertyColNameConverter,
    private val propertyValueMapper: PropertyValueMapper = PropertyValueMapperDefaultImpl
) : ReflectEntityMapper<T> {

    private var _autoGenerateId: Boolean
    private var _primaryProperty: KProperty1<T, *>
    private val _constructor: KFunction<T>
    private val _properties: List<KProperty1<T, *>>
    private val _parameters: List<KParameter>

    init {
        check(kClass.isData) { "$kClass is not a data class" }

        val properties = kClass.memberProperties.filter {
            it.javaField!!.getAnnotation(Ignore::class.java) == null
        }

        this._properties = properties

        val keys = properties.filter { it.javaField!!.getAnnotation(Id::class.java) != null }
        check(keys.isNotEmpty() && keys.size == 1) { "only One Id support" }
        this._primaryProperty = keys.first()
        this._autoGenerateId = this._primaryProperty.javaField!!.getAnnotation(GeneratedValue::class.java) != null


        val propertyNameSet = this._properties.map { it.name }.toSortedSet()
        val constructor = kClass.constructors.firstOrNull { constructor ->
            val params = constructor.parameters
            params.map { it.name }.containsAll(propertyNameSet)
        }

        checkNotNull(constructor) {
            "could not find a constructor contains all properties"
        }

        this._constructor = constructor

        this._parameters = constructor.parameters
    }


    override fun columns(): List<String> = this._properties.map { converter.propertyToColName(it.name) }
    override fun rowMapper(): (ResultSet) -> T {
        val propertyNameSet = this._properties.map { it.name }.toSortedSet()
        val associateBy = this._parameters.associateBy { it.name!! }
        return {
            val list = buildMap {
                for (param in _parameters) {
                    if (propertyNameSet.contains(param.name!!)) {
                        put(
                            associateBy[param.name!!]!!,
                            propertyValueMapper.getValueFromResultSet(
                                it, converter.propertyToColName(param.name!!), param.type.jvmErasure
                            )
                        )
                    }
                }
            }

            _constructor.callBy(list)

//            _constructor.call(*list.toTypedArray())

        }
    }

    override fun autoGenerateId(): Boolean = this._autoGenerateId
    override fun setPrimaryKeyValue(entity: T, pkValue: Number) {
        val prop = this._primaryProperty
        if (prop is KMutableProperty1<T, *>) {
            when (prop.setter.parameters[1].type.jvmErasure) {
                Int::class -> prop.setter.call(entity, pkValue.toInt())
                Long::class -> prop.setter.call(entity, pkValue.toLong())
                else -> prop.setter.call(entity, pkValue)
            }

        }

    }

    override fun primaryKeyColumnName(): String = this._primaryProperty.name


    override fun primaryKeyValue(entity: T): Any? = this._primaryProperty.get(entity)

    override fun allPropertyValues(entity: T, withPrimaryKey: Boolean): List<Any?> =
        if (withPrimaryKey)
            _properties.map { it.get(entity) }
        else _properties.asSequence()
            .filter { it.name != this.primaryKeyColumnName() }.map { it.get(entity) }.toList()
}