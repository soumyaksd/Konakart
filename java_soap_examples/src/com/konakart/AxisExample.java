//
// (c) 2006 DS Data Systems UK Ltd, All rights reserved.
//
// DS Data Systems and KonaKart and their respective logos, are 
// trademarks of DS Data Systems UK Ltd. All rights reserved.
//
// The information in this document is the proprietary property of
// DS Data Systems UK Ltd. and is protected by English copyright law,
// the laws of foreign jurisdictions, and international treaties,
// as applicable. No part of this document may be reproduced,
// transmitted, transcribed, transferred, modified, published, or
// translated into any language, in any form or by any means, for
// any purpose other than expressly permitted by DS Data Systems UK Ltd.
// in writing.
//

package com.konakart;

import com.konakart.app.EngineConfig;
import com.konakart.app.KKWSEng;
import com.konakart.appif.CountryIf;
import com.konakart.appif.CurrencyIf;
import com.konakart.appif.DataDescriptorIf;
import com.konakart.appif.EngineConfigIf;
import com.konakart.appif.KKEngIf;
import com.konakart.appif.ManufacturerIf;
import com.konakart.appif.ProductSearchIf;
import com.konakart.appif.ProductsIf;
import com.konakart.util.KKConstants;
import com.konakart.ws.KKWSEngIf;
import com.konakart.ws.KKWSEngIfServiceLocator;
import com.konakart.wsapp.Country;
import com.konakart.wsapp.Currency;
import com.konakart.wsapp.DataDescriptor;
import com.konakart.wsapp.Manufacturer;
import com.konakart.wsapp.ProductSearch;
import com.konakart.wsapp.Products;

/**
 * This is a simple example of an AXIS client accessing KonaKart services using SOAP
 * 
 * "usage: AxisExample"
 */
public final class AxisExample
{
    private AxisExample()
    {
    }

    static final String usage = "Usage: AxisExample";

    /**
     * @param args
     *            there are no arguments in fact
     */
    public static void main(String[] args)
    {
        if (args != null && args.length > 1)
        {
            System.out.println("Unrecognised arguments:\n" + usage);
            System.exit(1);
        }

        System.out.println("First the low-level version...\n");
        lowLevelVersion();
        
        System.out.println("\nNext the recommended high-level version...\n");
        highLevelVersion();
        
    }
    
    private static void lowLevelVersion()
    {
        try
        {
            KKWSEngIf port = new KKWSEngIfServiceLocator().getKKWebServiceEng();

            // make some example calls

            Country[] countries = port.getAllCountries();
            System.out.println("There are " + countries.length + " countries");

            Manufacturer[] manufacturers = port.getAllManufacturers();
            System.out.println("There are " + manufacturers.length + " manufacturers");

            Currency curr = port.getDefaultCurrency();
            System.out.println("The default currency is: " + curr.getCode());
            
            //String sessionId = port.login("doe@konakart.com", "password");
            String sessionId = null;
            
            DataDescriptor dataDesc = new DataDescriptor();
            dataDesc.setOffset(0);
            dataDesc.setLimit(100);
            dataDesc.setShowInvisible(true);
            // dataDesc.setCustom1("test");

            ProductSearch prodSearch = new ProductSearch();
            prodSearch.setManufacturerId(-100);              // -100 == Search all
            prodSearch.setCategoryId(-100);                  // -100 == Search all
            prodSearch.setSearchInSubCats(true);
            prodSearch.setSearchAllStores(false);
            prodSearch.setProductType(-1);                   // -1 == Not Set
            Products products = port.searchForProducts(sessionId, dataDesc, prodSearch, -1);

            System.out.println(products.getTotalNumProducts() + " products found");
            System.out.println(products.getProductArray().length + " length of product array");
            
            //com.konakartadmin.ws.KKWSAdminIf adminPort = new com.konakartadmin.ws.KKWSAdminIfServiceLocator().getKKWSAdmin();            
            //com.konakartadmin.app.AdminCustomer cust = adminPort.getCustomerForEmail(sessionId, "doe@konakart.com");
            //System.out.println("Customer = " + cust.toString());
            
        } catch (Exception e)
        {
            e.printStackTrace();
        }
    }
    
    private static void highLevelVersion()
    {
        try
        {
            EngineConfigIf engConfig = new EngineConfig();
            engConfig.setAppPropertiesFileName(KKConstants.KONAKART_APP_PROPERTIES_FILE);
            engConfig.setMode(EngineConfig.MODE_SINGLE_STORE);
            engConfig.setPropertiesFileName(KKConstants.KONAKART_PROPERTIES_FILE);
            engConfig.setStoreId(KKConstants.KONAKART_DEFAULT_STORE_ID);
            
            KKEngIf engine = new KKWSEng(engConfig);

            // make some example calls

            CountryIf[] countries = engine.getAllCountries();
            System.out.println("There are " + countries.length + " countries");

            ManufacturerIf[] manufacturers = engine.getAllManufacturers();
            System.out.println("There are " + manufacturers.length + " manufacturers");

            CurrencyIf curr = engine.getDefaultCurrency();
            System.out.println("The default currency is: " + curr.getCode());
            
            //String sessionId = port.login("doe@konakart.com", "password");
            String sessionId = null;
            
            DataDescriptorIf dataDesc = new com.konakart.app.DataDescriptor();
            dataDesc.setOffset(0);
            dataDesc.setLimit(100);
            dataDesc.setShowInvisible(true);
            // dataDesc.setCustom1("test");

            ProductSearchIf prodSearch = new com.konakart.app.ProductSearch();
            prodSearch.setManufacturerId(-100);              // -100 == Search all
            prodSearch.setCategoryId(-100);                  // -100 == Search all
            prodSearch.setSearchInSubCats(true);
            prodSearch.setSearchAllStores(false);
            prodSearch.setProductType(-1);                   // -1 == Not Set
            ProductsIf products = engine.searchForProducts(sessionId, dataDesc, prodSearch, -1);

            System.out.println(products.getTotalNumProducts() + " products found");
            System.out.println(products.getProductArray().length + " length of product array");
            
            //com.konakartadmin.ws.KKWSAdminIf adminPort = new com.konakartadmin.ws.KKWSAdminIfServiceLocator().getKKWSAdmin();            
            //com.konakartadmin.app.AdminCustomer cust = adminPort.getCustomerForEmail(sessionId, "doe@konakart.com");
            //System.out.println("Customer = " + cust.toString());
            
        } catch (Exception e)
        {
            e.printStackTrace();
        }
    }
}
