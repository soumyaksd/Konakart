//
// (c) 2006 DS Data Systems UK Ltd, All rights reserved.
//
// DS Data Systems and KonaKart and their respective logos, are 
// trademarks of DS Data Systems UK Ltd. All rights reserved.
//
// The information in this document is free software; you can redistribute 
// it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// 
// This software is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//

package com.konakart.apiexamples;

import com.konakart.app.DataDescriptor;
import com.konakart.app.EngineConfig;
import com.konakart.app.KKException;
import com.konakart.app.ProductSearch;
import com.konakart.appif.KKEngIf;
import com.konakart.appif.ProductIf;
import com.konakart.appif.ProductsIf;

/**
 * An example of how to retrieve a Product using the direct java engine and the RMI engine
 */
public class GetProduct extends BaseApiExample
{
    private static final String usage = "Usage: GetProduct\n" + COMMON_USAGE;

    /**
     * @param args
     */
    public static void main(String[] args)
    {
        parseArgs(args, usage, 0);

        try
        {
            /*
             * Get an instance of the KonaKart engine and login. The method called can be found in
             * BaseApiExample.java
             */
            EngineConfig engConf = new EngineConfig();
            engConf.setMode(getEngineMode());
            engConf.setStoreId(getStoreId());
            engConf.setCustomersShared(isCustomersShared());
            engConf.setProductsShared(isProductsShared());
            engConf.setCategoriesShared(isCategoriesShared());

            /*
             * Instantiate a direct java Engine by name - to find a product 
             */
            KKEngIf kkEng = getKKEngByName(KKEngName, engConf);
            System.out.println("\nGet a product using the KKEng engine");
            ProductIf prod = getProductUsingEngine(kkEng);
            System.out.println("Product Id        = " + prod.getId());
            System.out.println("Product Name      = " + prod.getName());
            System.out.println("Product CustomObj = " + (String)(prod.getCustomObj()));

            /*
             * Instantiate a SOAP Engine by name - to find a product 
             */
            KKEngIf soapEng = getKKEngByName(KKWSEngName, engConf);
            System.out.println("\nGet a product using the KKWSEng engine");
            prod = getProductUsingEngine(soapEng);
            System.out.println("Product Id        = " + prod.getId());
            System.out.println("Product Name      = " + prod.getName());
            System.out.println("Product CustomObj = " + (String)(prod.getCustomObj()));

            /*
             * Instantiate a java RMI Engine by name - to find a product
             */
            KKEngIf rmiEng = getKKEngByName(KKRMIEngName, engConf);
            System.out.println("\nGet a product using the KKRMIEng engine");
            prod = getProductUsingEngine(rmiEng);
            System.out.println("Product Id        = " + prod.getId());
            System.out.println("Product Name      = " + prod.getName());
            System.out.println("Product CustomObj = " + (String)(prod.getCustomObj()));

            /*
             * Instantiate a java JSON Engine by name - to find a product 
             */
            KKEngIf jsonEng = getKKEngByName(KKJSONEngName, engConf);
            System.out.println("\nGet a product using the KKJSONEng engine");
            prod = getProductUsingEngine(jsonEng);
            System.out.println("Product Id        = " + prod.getId());
            System.out.println("Product Name      = " + prod.getName());
            System.out.println("Product CustomObj = " + (String)(prod.getCustomObj()));

            /*
             * Instantiate a java JAXWS Engine by name - to find a product 
             */
            KKEngIf jaxwsEng = getKKEngByName(KKJAXWSEngName, engConf);
            System.out.println("\nGet a product using the KKJAXWSEng engine");
            prod = getProductUsingEngine(jaxwsEng);
            System.out.println("Product Id        = " + prod.getId());
            System.out.println("Product Name      = " + prod.getName());
            System.out.println("Product CustomObj = " + (String)(prod.getCustomObj()));

            System.out.println("Completed Successfully");

        } catch (Exception e)
        {
            e.printStackTrace();
        }
    }

    private static ProductIf getProductUsingEngine(KKEngIf kkEng) throws KKException
    {
        // Create a product search object
        ProductSearch ps = new ProductSearch();
        ps.setSearchText("Mary");

        // Create a data descriptor
        DataDescriptor datadesc = new DataDescriptor();
        datadesc.setOffset(0);
        datadesc.setLimit(1);

        // Now we can search for the products.
        ProductsIf prods = kkEng.searchForProducts(/* sessionId */null, datadesc, ps,
                DEFAULT_LANGUAGE);

        if (prods.getProductArray().length == 1)
        {
            System.out.println("Search Found 1 Product, now retrieve the whole product");
            ProductIf prod = kkEng.getProduct(/* sessionId */null, prods.getProductArray()[0].getId(), 
                    DEFAULT_LANGUAGE);
            // Now add our own object to the KonaKart Product
            prod.setCustomObj("This can be any object, but this is just a String");
            return prod;
        } else if (prods.getProductArray().length == 0)
        {
            System.out.println("No Products Found Matching Search Criteria");
        } else
        {
            System.out.println(prods.getProductArray().length
                    + " Products Found but only one was expected");
        }

        return null;
    }
}
