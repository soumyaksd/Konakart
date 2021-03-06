package com.konakart.app;

import com.konakart.appif.*;

/**
 *  The KonaKart Custom Engine - GetProductsPerCategoryPerManufacturer - Generated by CreateKKCustomEng
 */
@SuppressWarnings("all")
public class GetProductsPerCategoryPerManufacturer
{
    KKEng kkEng = null;

    /**
     * Constructor
     */
     public GetProductsPerCategoryPerManufacturer(KKEng _kkEng)
     {
         kkEng = _kkEng;
     }

     public ProductsIf getProductsPerCategoryPerManufacturer(String sessionId, DataDescriptorIf dataDesc, int categoryId, int manufacturerId, int languageId) throws KKException
     {
         return kkEng.getProductsPerCategoryPerManufacturer(sessionId, dataDesc, categoryId, manufacturerId, languageId);
     }
}
